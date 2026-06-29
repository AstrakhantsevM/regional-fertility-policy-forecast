import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from seaborn.matrix import ClusterGrid
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from typing import List, Optional, Tuple

# ========================
# DATA LOADING: Excel loading utilities
# ========================

def load_sheets(path: str) -> dict:
    all_sheets = pd.read_excel(path, sheet_name=None)
    
    filtered_sheets = {
        sheet_name: df
        for sheet_name, df in all_sheets.items()
        if 'не добавлять' not in [str(col).lower() for col in df.columns]
    }
    
    return filtered_sheets

# ========================
# DATA ENGENEERING
# ========================

def get_relative_metric(df_numerator: pd.DataFrame, df_denominator: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет относительный показатель (числитель / знаменатель) для панельных данных.
    
    Параметры:
    df_numerator (pd.DataFrame): Датафрейм-числитель (например, РМК).
    df_denominator (pd.DataFrame): Датафрейм-знаменатель (например, среднедушевые доходы).
    
    Возвращает:
    pd.DataFrame: Новый датафрейм той же структуры ('Регион' + столбцы-года) с результатами деления.
    """
    
    # Временно делаем 'Регион' индексом, чтобы Pandas мог точно сопоставить строки
    num_idx = df_numerator.set_index('Регион')
    den_idx = df_denominator.set_index('Регион')
    
    # Метод .div() безопасно делит значения. 
    # Pandas сам найдет одинаковые названия столбцов (года) и индексов (регионы).
    # Там, где данные не пересекаются (например, год есть в одном df, но нет в другом), появится NaN.
    relative_df = num_idx.div(den_idx)
    
    # Сбрасываем индекс, чтобы 'Регион' снова стал обычной первой колонкой
    relative_df = relative_df.reset_index()
    
    return relative_df


# ========================
# FORMATTING TO LONG FORMAT
# ========================

def to_long(df, value_name, id_col='Регион', exclude_patterns=None):
    """
    Преобразует таблицу с региональными данными из широкого формата (wide) 
    в длинный панельный формат (long). Сначала очищает данные, затем разворачивает их.

    Параметры:
        df (pd.DataFrame): Исходный датафрейм, где столбцы — это годы (например, '2020', '2021'), 
                           а строки — субъекты/регионы.
        value_name (str): Название нового столбца, в который будут записаны значения показателя 
                          (например, 'tfr', 'unemployment').
        id_col (str, по умолчанию 'Регион'): Название столбца в исходной таблице, 
                                             в котором хранятся названия регионов.
        exclude_patterns (list of str, опционально): Список регулярных выражений (regex) для 
                                                     исключения ненужных строк. Например, 
                                                     ['Россия', 'ФО$'] уберет агрегаты по РФ и округам.

    Возвращает:
        pd.DataFrame: Очищенный датафрейм в длинном формате со столбцами: ['region', 'year', value_name].
                      Столбец с регионом всегда будет переименован в 'region' для удобства джойна.
    """
    
    # --- Шаг 1: Очистка данных ДО разворачивания (для экономии памяти и времени) ---
    
    # Создаем копию без строк, где название региона отсутствует (NaN)
    df_clean = df.dropna(subset=[id_col]).copy()
    
    # Приводим названия регионов к строковому типу и удаляем пробелы по краям (частая проблема Excel)
    df_clean[id_col] = df_clean[id_col].astype(str).str.strip()
    
    # Исключаем строки, которые после удаления пробелов оказались пустыми
    df_clean = df_clean[df_clean[id_col] != '']

    # --- Шаг 2: Исключение агрегатных строк (если переданы паттерны) ---
    if exclude_patterns:
        # Склеиваем список паттернов в одну строку через логическое ИЛИ ('|')
        pattern = '|'.join(exclude_patterns)
        
        # Оставляем только те строки, где название региона НЕ содержит паттерн
        # na=False предотвращает ошибки, если попадутся пустые значения
        df_clean = df_clean[~df_clean[id_col].str.contains(pattern, case=False, na=False)]

    # --- Шаг 3: Разворачивание таблицы (Wide -> Long) ---
    
    # Собираем список столбцов, названия которых состоят только из цифр (это и есть наши годы)
    year_cols = [col for col in df_clean.columns if str(col).isdigit()]
    
    # Функция melt преобразует столбцы с годами в две колонки: 'year' и колонку со значениями
    long = df_clean.melt(
        id_vars=id_col,          # Столбец-идентификатор, который остается неизменным
        value_vars=year_cols,    # Столбцы, которые нужно свернуть (годы)
        var_name='year',         # Как назвать новый столбец с названиями свернутых колонок
        value_name=value_name    # Как назвать новый столбец со значениями
    )

    # --- Шаг 4: Финальное форматирование ---
    
    # Приводим столбец с годами к целочисленному типу (int), чтобы графики и сортировки работали корректно
    long['year'] = long['year'].astype(int)
    
    # Переименовываем исходный столбец с регионами в стандартизированное имя 'region'
    long = long.rename(columns={id_col: 'region'})

    return long

# ================================================
#    АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ (ВИЗУАЛИЗАЦИЯ
# ================================================

def _create_custom_cmap() -> LinearSegmentedColormap:
    """
    Создает кастомную цветовую палитру для тепловой карты пропусков.

    Логика шкалы:
    - 0.0  -> светло-зеленый, пропусков нет;
    - >0.0 -> переход в желтый;
    - 0.5  -> красно-оранжевый;
    - 1.0  -> темно-бордовый.

    Returns
    -------
    LinearSegmentedColormap
        Цветовая палитра для heatmap.
    """
    # Здесь задаем опорные точки цветового градиента.
    # Первая координата в кортеже — это положение цвета на шкале [0, 1].
    colors = [
        (0.0, '#e5f5e0'),   # Светло-зеленый: полная заполненность
        (0.01, '#fff7bc'),  # Очень светло-желтый: появился хотя бы минимальный пропуск
        (0.5, '#fc4e2a'),   # Красно-оранжевый: заметная доля пропусков
        (1.0, '#800026')    # Темно-бордовый: экстремально высокая доля пропусков
    ]

    # Формируем непрерывную палитру на 256 оттенков.
    return LinearSegmentedColormap.from_list(
        "custom_missingness",
        colors,
        N=256
    )


def _prepare_missingness_matrix(
    df: pd.DataFrame,
    year_col: str,
    ignore_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Готовит матрицу долей пропусков по годам.

    На выходе получается прямоугольная матрица:
    - строки: анализируемые переменные,
    - столбцы: годы,
    - значения: доля пропусков в данном году.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    year_col : str
        Название столбца с годом.
    ignore_cols : Optional[List[str]]
        Список столбцов, которые нужно исключить из анализа
        (например, region_id, region_name и другие технические поля).

    Returns
    -------
    pd.DataFrame
        Матрица долей пропусков по годам.
    """
    if ignore_cols is None:
        ignore_cols = []

    # Формируем единый список колонок, которые надо исключить.
    cols_to_drop = [year_col] + ignore_cols

    # Оставляем только те переменные, по которым мы действительно хотим
    # анализировать пропуски.
    metrics_cols = [col for col in df.columns if col not in cols_to_drop]

    # Считаем долю пропусков по каждому году.
    # Внутри каждой группы по году:
    # - x.isna() превращает значения в True/False,
    # - mean() по булевым значениям дает долю пропусков.
    missing_data = (
        df.groupby(year_col)[metrics_cols]
        .apply(lambda x: x.isna().mean())
        .T
    )

    return missing_data


def _build_row_linkage_with_global_level_spacing(
    matrix: pd.DataFrame,
    method: str = 'average',
    metric: str = 'euclidean',
    min_leaf_distance_share: float = 0.10,
    min_level_gap_share: float = 0.05,
    transform_power: float = 0.80
) -> np.ndarray:
    """
    Строит linkage-матрицу для строк и перераспределяет высоты дендрограммы так,
    чтобы дерево было визуально читаемым по всей глубине, а не только у heatmap.

    Ключевая идея:
    ----------------
    В стандартной дендрограмме короткие расстояния часто "налипают" к heatmap,
    а соседние уровни слияния могут выглядеть почти одинаковыми по длине.
    Тогда дерево визуально сплющивается.

    Чтобы исправить это, мы:
    1. Считаем исходный linkage.
    2. Берем столбец Z[:, 2], который хранит высоты слияний.
    3. Нормируем эти высоты в [0, 1].
    4. Мягко растягиваем нижнюю и среднюю часть шкалы степенным преобразованием.
    5. Дополнительно задаем:
       - минимальную дистанцию для самого нижнего уровня,
       - минимальный шаг между любыми соседними уровнями.
    6. Восстанавливаем монотонность heights, потому что dendrogram ожидает
       неубывающие значения Z[:, 2].

    Parameters
    ----------
    matrix : pd.DataFrame
        Матрица для кластеризации строк.
    method : str
        Метод linkage для scipy.cluster.hierarchy.linkage.
    metric : str
        Метрика расстояния для scipy.spatial.distance.pdist.
    min_leaf_distance_share : float
        Минимальная длина самых коротких ветвей как доля от максимальной высоты.
        Например, 0.10 означает, что нижние ветви не будут визуально короче,
        чем 10% от полной высоты дендрограммы.
    min_level_gap_share : float
        Минимальный зазор между соседними уровнями слияния как доля от максимальной высоты.
        Именно этот параметр отвечает за то, чтобы сплющивание не возникало
        и у внутренних ветвей, а не только у самых правых.
    transform_power : float
        Степенное преобразование нормированных высот.
        Значения < 1 растягивают нижнюю часть шкалы сильнее.
        Значение 1.0 соответствует почти линейной шкале.

    Returns
    -------
    np.ndarray
        Модифицированная linkage-матрица для передачи в row_linkage.
    """
    # Сначала считаем попарные расстояния между строками матрицы.
    # Каждая строка — это одна переменная, описанная своим профилем пропусков по годам.
    pairwise_distances = pdist(matrix.values, metric=metric)

    # Строим стандартную linkage-матрицу.
    # Формат linkage:
    # - первые два столбца: какие кластеры объединяются,
    # - третий столбец: расстояние между ними,
    # - четвертый столбец: размер нового кластера.
    row_linkage = linkage(pairwise_distances, method=method).copy()

    # Извлекаем столбец высот слияния.
    raw_heights = row_linkage[:, 2]

    # Если дерево фактически не строится, возвращаем как есть.
    if len(raw_heights) == 0:
        return row_linkage

    raw_min = raw_heights.min()
    raw_max = raw_heights.max()

    # Если все высоты одинаковы, перераспределять нечего.
    if np.isclose(raw_min, raw_max):
        return row_linkage

    # Нормируем высоты в диапазон [0, 1].
    # Это нужно, чтобы все дальнейшие преобразования были масштабно-независимыми.
    normalized_heights = (raw_heights - raw_min) / (raw_max - raw_min)

    # Степенное преобразование:
    # - при transform_power < 1 нижняя часть шкалы растягивается сильнее,
    # - это помогает "вытащить" короткие ветви из правого края дендрограммы.
    transformed_heights = normalized_heights ** transform_power

    # Выделяем уникальные уровни слияния.
    # Логика минимального межуровневого зазора задается именно на уровне уникальных heights.
    unique_levels = np.unique(transformed_heights)
    unique_levels.sort()

    # В этом словаре мы построим отображение:
    # исходный уровень -> новый уровень после визуального растяжения.
    level_mapping = {}

    # Самый нижний уровень сразу ставим не ближе, чем min_leaf_distance_share.
    previous_level = min_leaf_distance_share

    for level_index, level_value in enumerate(unique_levels):
        if level_index == 0:
            # Первый уровень не должен прилипать к heatmap.
            new_level = max(level_value, min_leaf_distance_share)
        else:
            # Каждый следующий уровень должен быть:
            # 1) не меньше своего исходного растянутого значения,
            # 2) не меньше предыдущего уровня + минимальный шаг.
            new_level = max(level_value, previous_level + min_level_gap_share)

        level_mapping[level_value] = new_level
        previous_level = new_level

    # Перекладываем отображение на все heights linkage-матрицы.
    remapped_heights = np.array([
        level_mapping[level_value] for level_value in transformed_heights
    ])

    # Если после навязывания шагов максимум вышел за предел 1.0,
    # аккуратно перенормируем обратно в [min_leaf_distance_share, 1.0].
    remapped_max = remapped_heights.max()

    if remapped_max > 1.0:
        remapped_min = remapped_heights.min()

        # Защита от теоретически вырожденного случая.
        if not np.isclose(remapped_max, remapped_min):
            remapped_heights = min_leaf_distance_share + (
                (remapped_heights - remapped_min) / (remapped_max - remapped_min)
            ) * (1.0 - min_leaf_distance_share)

    # Для dendrogram heights должны быть монотонно неубывающими.
    # Иначе возможны пересечения или некорректная геометрия ветвей.
    remapped_heights = np.maximum.accumulate(remapped_heights)

    # Возвращаемся к исходному масштабу высот.
    adjusted_heights = remapped_heights * raw_max

    # Подменяем столбец расстояний в linkage-матрице.
    row_linkage[:, 2] = adjusted_heights

    return row_linkage


def _set_centered_year_ticks(
    g: ClusterGrid,
    years: List[int],
    start_year: int,
    label_step: int
) -> None:
    """
    Настраивает ось X heatmap так, чтобы подписи годов были строго по центру ячеек.

    Parameters
    ----------
    g : ClusterGrid
        Объект seaborn clustermap.
    years : List[int]
        Список годов в том порядке, в котором они отображаются на heatmap.
    start_year : int
        Базовый год для логики шага подписей.
    label_step : int
        Через сколько лет выводить подпись.

    Returns
    -------
    None
    """
    # У heatmap каждая ячейка занимает интервал [i, i+1],
    # значит центр ячейки расположен в i + 0.5.
    tick_positions = [i + 0.5 for i in range(len(years))]

    # Подписываем не все годы, а только годы с заданным шагом.
    tick_labels = [
        str(year) if (year - start_year) % label_step == 0 else ""
        for year in years
    ]

    g.ax_heatmap.set_xticks(tick_positions)
    g.ax_heatmap.set_xticklabels(
        tick_labels,
        rotation=0,
        ha='center'
    )

    # Делаем тики более заметными, чтобы подписи лучше читались.
    g.ax_heatmap.tick_params(
        axis='x',
        bottom=True,
        labelbottom=True,
        pad=6,
        length=6,
        direction='out'
    )


def _align_row_dendrogram_to_heatmap(
    g: ClusterGrid,
    dendrogram_width_ratio: float = 0.08,
    gap: float = 0.0
) -> None:
    """
    Подгоняет ось дендрограммы строк вплотную к heatmap.

    Важно:
    -------
    Если просто сжимать дендрограмму по width, не контролируя ее правую границу,
    появится зазор между ветвями и heatmap. Поэтому мы задаем позицию оси так,
    чтобы правый край дендрограммы совпадал с левым краем heatmap.

    Parameters
    ----------
    g : ClusterGrid
        Объект seaborn clustermap.
    dendrogram_width_ratio : float
        Желаемая ширина области дендрограммы в координатах figure.
    gap : float
        Контролируемый зазор между дендрограммой и heatmap.
        Если нужен контакт "вплотную", используйте 0.0.

    Returns
    -------
    None
    """
    # Получаем текущие позиции осей.
    dendro_pos = g.ax_row_dendrogram.get_position()
    heatmap_pos = g.ax_heatmap.get_position()

    # Правая граница дендрограммы должна совпадать с левой границей heatmap
    # с учетом возможного маленького зазора gap.
    new_right_edge = heatmap_pos.x0 - gap
    new_width = dendrogram_width_ratio
    new_x0 = new_right_edge - new_width

    # Переставляем ось дендрограммы.
    g.ax_row_dendrogram.set_position([
        new_x0,
        dendro_pos.y0,
        new_width,
        dendro_pos.height
    ])

    # Убираем тики и рамки — они здесь не нужны и могут визуально шуметь.
    g.ax_row_dendrogram.set_xticks([])
    g.ax_row_dendrogram.set_yticks([])

    for spine in g.ax_row_dendrogram.spines.values():
        spine.set_visible(False)


def _plot_final_missingness_map(
    df: pd.DataFrame,
    year_col: str,
    ignore_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 9),
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    label_step: int = 3,
    dendrogram_width_ratio: float = 0.08,
    dendrogram_gap: float = 0.0,
    linkage_method: str = 'average',
    linkage_metric: str = 'euclidean',
    min_leaf_distance_share: float = 0.10,
    min_level_gap_share: float = 0.05,
    transform_power: float = 0.80
) -> None:
    """
    Строит итоговую карту пропусков с дендрограммой по строкам.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    year_col : str
        Название столбца с годом.
    ignore_cols : Optional[List[str]]
        Столбцы, которые исключаются из анализа пропусков.
    figsize : Tuple[int, int]
        Размер фигуры.
    start_year : Optional[int]
        Начальный год для отображения подписей.
        Если None, берется минимальный год из данных.
    end_year : Optional[int]
        Конечный год для отображения подписей.
        Сейчас используется только как внешний параметр интерфейса;
        фактическая ось строится по годам, присутствующим в матрице.
    label_step : int
        Через сколько лет отображать подпись на оси X.
    dendrogram_width_ratio : float
        Ширина области дендрограммы.
    dendrogram_gap : float
        Зазор между дендрограммой и heatmap.
    linkage_method : str
        Метод иерархической кластеризации.
    linkage_metric : str
        Метрика расстояния между строками.
    min_leaf_distance_share : float
        Минимальная длина нижних ветвей.
    min_level_gap_share : float
        Минимальный шаг между соседними уровнями дерева.
    transform_power : float
        Параметр нелинейного растяжения шкалы высот.

    Returns
    -------
    None
    """
    # Шаг 1. Готовим матрицу долей пропусков.
    missing_data = _prepare_missingness_matrix(
        df=df,
        year_col=year_col,
        ignore_cols=ignore_cols
    )

    # Если матрица пустая, строить нечего.
    if missing_data.empty:
        print("В данных вообще нет пропусков.")
        return

    # Получаем фактический порядок годов из данных.
    years_in_data = list(missing_data.columns)

    # Если стартовый год не задан, берем минимум из данных.
    if start_year is None:
        start_year = min(years_in_data)

    # Если конечный год не задан, берем максимум из данных.
    if end_year is None:
        end_year = max(years_in_data)

    # Шаг 2. Строим linkage вручную, чтобы контролировать высоты дерева.
    row_linkage = _build_row_linkage_with_global_level_spacing(
        matrix=missing_data,
        method=linkage_method,
        metric=linkage_metric,
        min_leaf_distance_share=min_leaf_distance_share,
        min_level_gap_share=min_level_gap_share,
        transform_power=transform_power
    )

    # Шаг 3. Строим clustermap.
    # Важно: row_linkage передаем явно, чтобы seaborn не пересчитывал дерево сам.
    g = sns.clustermap(
        missing_data,
        row_cluster=True,
        col_cluster=False,
        row_linkage=row_linkage,
        cmap=_create_custom_cmap(),
        figsize=figsize,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='white',
        cbar_pos=(1.02, 0.15, 0.02, 0.6),
        tree_kws={'linewidths': 1.2},
        dendrogram_ratio=(0.08, 0.05)
    )

    # Шаг 4. Подгоняем дендрограмму вплотную к heatmap.
    _align_row_dendrogram_to_heatmap(
        g=g,
        dendrogram_width_ratio=dendrogram_width_ratio,
        gap=dendrogram_gap
    )

    # Шаг 5. Оформляем заголовок.
    g.fig.suptitle(
        'Структура пропусков данных по годам (Зеленый = 100% заполнено)\nЛинии отвечают за группировку по структуре пропусков',
        y=1.00,
        fontsize=16
    )

    # Шаг 6. Подписываем ось X.
    g.ax_heatmap.set_xlabel(
        f'',
        fontsize=12
    )

    # Шаг 7. Центрируем подписи годов под ячейками.
    _set_centered_year_ticks(
        g=g,
        years=years_in_data,
        start_year=start_year,
        label_step=label_step
    )

    # Финальный рендер.
    plt.show()


def plot_missingness(
    df: pd.DataFrame,
    year_col: str = 'year',
    technical_columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 9),
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    label_step: int = 3,
    dendrogram_width_ratio: float = 0.08,
    dendrogram_gap: float = 0.0,
    linkage_method: str = 'average',
    linkage_metric: str = 'euclidean',
    min_leaf_distance_share: float = 0.10,
    min_level_gap_share: float = 0.05,
    transform_power: float = 0.80
) -> None:
    """
    Оркестратор построения итоговой карты пропусков.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    year_col : str
        Название столбца с годом.
    technical_columns : Optional[List[str]]
        Технические столбцы, которые не анализируются как переменные.
    figsize : Tuple[int, int]
        Размер фигуры.
    start_year : Optional[int]
        Начальный год для подписей на оси X.
    end_year : Optional[int]
        Конечный год для подписей на оси X.
    label_step : int
        Шаг отображения подписей годов.
    dendrogram_width_ratio : float
        Ширина области дендрограммы.
    dendrogram_gap : float
        Зазор между дендрограммой и heatmap.
    linkage_method : str
        Метод linkage.
    linkage_metric : str
        Метрика расстояния для linkage.
    min_leaf_distance_share : float
        Минимальная длина самых коротких ветвей.
    min_level_gap_share : float
        Минимальный шаг между уровнями дерева.
    transform_power : float
        Параметр нелинейного растяжения heights.

    Returns
    -------
    None
    """
    # Если технические колонки не заданы, используем типовой набор.
    if technical_columns is None:
        technical_columns = ['region']

    # Передаем все параметры в основную функцию построения.
    _plot_final_missingness_map(
        df=df,
        year_col=year_col,
        ignore_cols=technical_columns,
        figsize=figsize,
        start_year=start_year,
        end_year=end_year,
        label_step=label_step,
        dendrogram_width_ratio=dendrogram_width_ratio,
        dendrogram_gap=dendrogram_gap,
        linkage_method=linkage_method,
        linkage_metric=linkage_metric,
        min_leaf_distance_share=min_leaf_distance_share,
        min_level_gap_share=min_level_gap_share,
        transform_power=transform_power
    )

# ================================================
#    Сдвиг на N лагов в панельном датафрейме
# ================================================

def shift_panel_by_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    exclude_cols: list[str],
    default_lag: int = 1,
    special_col: str = None,
    special_lag: int = None
) -> pd.DataFrame:
    """
    Сдвигает столбцы внутри групп с гибкими лагами:
    - все кроме exclude_cols и special_col: default_lag
    - special_col: special_lag (если задан)
    - exclude_cols: не сдвигаются

    Параметры
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    group_col : str
        Столбец группировки (region).
    time_col : str
        Столбец времени (year).
    exclude_cols : list[str]
        Столбцы без лага (region, year, СКР).
    default_lag : int, default=1
        Лаг для всех остальных столбцов.
    special_col : str, optional
        Столбец с особым лагом.
    special_lag : int, optional
        Лаг для special_col.

    Возвращает
    ----------
    pd.DataFrame
        Датафрейм с лагированием.
    """
    exclude_cols = list(exclude_cols)
    
    def _lag_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()
        
        for col in g.columns:
            if col in exclude_cols:
                continue  # Не сдвигаем
            elif col == special_col:
                g[col] = g[col].shift(special_lag)  # Особый лаг
            else:
                g[col] = g[col].shift(default_lag)  # Общий лаг
                
        return g
    
    return df.groupby(group_col, group_keys=False).apply(_lag_group, include_groups=False)