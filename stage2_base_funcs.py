import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
import seaborn as sns
from typing import List, Tuple, Any, Optional, Dict, Iterable
from matplotlib.colors import ListedColormap
import statsmodels.api as sm

# ========================
#  ЗАГРУЗКА ЛИСТОВ ЭКСЕЛЬ
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
#  ВИЗУАЛИЗАЦИЯ ПРОПУСКОВ
# ========================

def _create_binary_cmap() -> ListedColormap:
    """
    Создает бинарную цветовую палитру для карты пропусков
    в стиле основной работы.

    Логика шкалы:
    - 0 -> светло-зеленый, значение присутствует;
    - 1 -> темно-бордовый, значение пропущено.

    Returns
    -------
    ListedColormap
        Бинарная цветовая палитра для heatmap.
    """
    # Цвета берутся из ранее использованной палитры в основной работе:
    # - светло-зеленый для полной заполненности,
    # - темно-бордовый для пропусков.
    return ListedColormap([
        '#e5f5e0',  # Значение есть
        '#800026'   # Пропуск
    ])


def _extract_year_columns(
    df: pd.DataFrame,
    region_col: str
) -> Tuple[List[Any], List[int], int, int]:
    """
    Извлекает годовые колонки из wide-таблицы и сохраняет их
    в исходном формате имен столбцов.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    region_col : str
        Название столбца с регионом.

    Returns
    -------
    Tuple[List[Any], List[int], int, int]
        - Список реальных годовых колонок;
        - Список годов в int-виде;
        - Минимальный год;
        - Максимальный год.
    """
    year_columns = []
    year_values = []

    for col in df.columns:
        if col == region_col:
            continue

        col_as_str = str(col).strip()

        if col_as_str.isdigit():
            year_columns.append(col)
            year_values.append(int(col_as_str))

    if not year_columns:
        raise ValueError(
            "Не удалось найти годовые колонки. "
            "Проверьте, что названия столбцов с годами состоят из цифр."
        )

    sorted_pairs = sorted(zip(year_columns, year_values), key=lambda x: x[1])

    sorted_year_columns = [pair[0] for pair in sorted_pairs]
    sorted_year_values = [pair[1] for pair in sorted_pairs]

    year_min = min(sorted_year_values)
    year_max = max(sorted_year_values)

    return sorted_year_columns, sorted_year_values, year_min, year_max


def _calculate_missing_stats(
    df: pd.DataFrame,
    region_col: str,
    year_columns: List[Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Вычисляет матрицу пропусков и выделяет регионы,
    в которых есть хотя бы один пропуск.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    region_col : str
        Название столбца с регионом.
    year_columns : List[Any]
        Список годовых колонок.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, int]
        - Матрица пропусков;
        - Датафрейм только с проблемными регионами;
        - Количество проблемных регионов.
    """
    # True = пропуск, False = значение есть.
    missing_matrix = df.set_index(region_col)[year_columns].isna()

    # Определяем регионы, где есть хотя бы один пропуск.
    regions_with_missing_mask = missing_matrix.any(axis=1)
    regions_with_missing = regions_with_missing_mask[regions_with_missing_mask].index

    df_missing_only = df[df[region_col].isin(regions_with_missing)].copy()
    count_missing_regions = int(regions_with_missing_mask.sum())

    return missing_matrix, df_missing_only, count_missing_regions

def _set_x_ticks(
    ax: plt.Axes,
    year_values: List[int],
    start_year_for_labels: int,
    end_year_for_labels: Optional[int],
    label_step: int
) -> None:
    """
    Настраивает подписи годов на оси X.

    Логика:
    -------
    Подписываются только те годы, которые:
    - присутствуют в данных,
    - не меньше start_year_for_labels,
    - не больше end_year_for_labels (если он задан),
    - попадают в шаг label_step.

    Parameters
    ----------
    ax : plt.Axes
        Ось matplotlib.
    year_values : List[int]
        Список годов в порядке отображения на heatmap.
    start_year_for_labels : int
        Год, от которого начинается шаг подписей.
    end_year_for_labels : Optional[int]
        Верхняя граница подписей. Если None, используется максимум из данных.
    label_step : int
        Шаг подписей по оси X.

    Returns
    -------
    None
    """
    if end_year_for_labels is None:
        end_year_for_labels = max(year_values)

    tick_positions = [i + 0.5 for i in range(len(year_values))]
    tick_labels = []

    for year in year_values:
        if (
            year >= start_year_for_labels and
            year <= end_year_for_labels and
            (year - start_year_for_labels) % label_step == 0
        ):
            tick_labels.append(str(year))
        else:
            tick_labels.append("")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')


def _plot_region_missing_heatmap(
    df_missing_only: pd.DataFrame,
    region_col: str,
    year_columns: List[Any],
    year_values: List[int],
    figsize: Tuple[float, float] = (12, 8),
    start_year_for_labels: int = 1990,
    end_year_for_labels: Optional[int] = 2026,
    label_step: int = 3
) -> None:
    """
    Строит бинарную карту пропусков по регионам.

    Логика:
    -------
    - 0 = значение присутствует;
    - 1 = значение пропущено.

    Parameters
    ----------
    df_missing_only : pd.DataFrame
        Подтаблица только с проблемными регионами.
    region_col : str
        Название столбца с регионом.
    year_columns : List[Any]
        Список годовых колонок.
    year_values : List[int]
        Список годов в числовом формате.
    figsize : Tuple[float, float]
        Базовый размер фигуры.
    start_year_for_labels : int
        Год, от которого начинается отображение подписей.
    end_year_for_labels : Optional[int]
        Последний год, который допускается показывать на оси X.
    label_step : int
        Шаг подписей по оси X.

    Returns
    -------
    None
    """
    # Формируем бинарную матрицу:
    # 0 = значение есть,
    # 1 = пропуск.
    heatmap_data = (
        df_missing_only
        .set_index(region_col)[year_columns]
        .isna()
        .astype(int)
    )

    # Высоту подстраиваем под число проблемных регионов,
    # чтобы горизонтальные подписи не наслаивались.
    dynamic_height = max(figsize[1], len(heatmap_data) * 0.35)

    plt.figure(figsize=(figsize[0], dynamic_height))

    ax = sns.heatmap(
        heatmap_data,
        cmap=_create_binary_cmap(),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor='white'
    )

    # Настраиваем подписи годов с заданным шагом.
    _set_x_ticks(
        ax=ax,
        year_values=year_values,
        start_year_for_labels=start_year_for_labels,
        end_year_for_labels=end_year_for_labels,
        label_step=label_step
    )

    # Регионы оставляем горизонтально.
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.xlabel('')
    plt.ylabel('')
    plt.title(
        'Карта пропусков по регионам\n'
        'Светло-зеленый = значение есть, темно-бордовый = пропуск',
        fontsize=13
    )
    plt.tight_layout()
    plt.show()


def analyze_missing_data(
    df: pd.DataFrame,
    region_col: str = 'Регион',
    figsize: Tuple[float, float] = (12, 8),
    start_year_for_labels: int = 1990,
    end_year_for_labels: Optional[int] = 2027,
    label_step: int = 3
) -> None:
    """
    Оркестратор анализа пропусков в wide-таблице СКР по регионам.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм в wide-формате.
    region_col : str
        Название столбца с регионом.
    figsize : Tuple[float, float]
        Базовый размер фигуры.
    start_year_for_labels : int
        Год, от которого начинается отображение подписей по оси X.
    end_year_for_labels : Optional[int]
        Верхняя граница подписей по оси X.
    label_step : int
        Шаг подписей по оси X.

    Returns
    -------
    None
    """
    # Шаг 1. Извлекаем годовые колонки.
    year_columns, year_values, year_min, year_max = _extract_year_columns(
        df=df,
        region_col=region_col
    )

    # Шаг 2. Считаем статистику пропусков.
    missing_matrix, df_missing_only, count_missing_regions = _calculate_missing_stats(
        df=df,
        region_col=region_col,
        year_columns=year_columns
    )

    # Шаг 3. Если пропусков нет, визуализацию не строим.
    if count_missing_regions == 0:
        print("Пропуски не обнаружены. Данные полны по всем субъектам.")
        return

    # Шаг 4. Строим карту пропусков.
    _plot_region_missing_heatmap(
        df_missing_only=df_missing_only,
        region_col=region_col,
        year_columns=year_columns,
        year_values=year_values,
        figsize=figsize,
        start_year_for_labels=start_year_for_labels,
        end_year_for_labels=end_year_for_labels,
        label_step=label_step
    )

# ========================
# WIDE TO LONG FORMATTING
# ========================

def transform_to_long(df, region_col='Регион'):
    """
    Переводит wide-датасет (годы в колонках) в long-формат.
    """
    # Определяем колонки-годы (все, что можно привести к числу)
    year_cols = [col for col in df.columns if str(col).isdigit()]
    
    df_long = df.melt(
        id_vars=region_col, 
        value_vars=year_cols, 
        var_name='Year', 
        value_name='TFR'
    )
    
    # Приведение типов и сортировка
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['TFR'] = pd.to_numeric(df_long['TFR'], errors='coerce')
    
    # Удаляем явные пропуски, чтобы не мешать расчетам скользящих средних
    df_long = df_long.dropna(subset=['TFR']).sort_values([region_col, 'Year'])
    
    return df_long

