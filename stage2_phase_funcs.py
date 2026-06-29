"""
Модуль для назначения фаз демографических переходов и сенситив-анализа.

Основан на адаптированной методологии ООН:
  - Фаза 2: период снижения СКР до точки стабилизации
  - Фаза 3: период после стабилизации

Конфигурация алгоритма централизована в PhaseConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Константы по умолчанию для имён колонок
# ---------------------------------------------------------------------------
_COL_REGION = "Регион"
_COL_YEAR   = "Year"
_COL_TFR    = "TFR"
_COL_PHASE  = "Phase"

# Год введения маткапитала — граница fallback-поиска минимума
_MATERNAL_CAPITAL_YEAR = 2007

# ---------------------------------------------------------------------------
# Конфигурация алгоритма
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseConfig:
    """
    Параметры алгоритма назначения фаз демографического перехода.

    Атрибуты
    --------
    smoothing_window : int
        Размер окна сглаживания (w). Минимум 1.
    forward_window : int
        Окно подтверждения стабилизации по методологии ООН (k). Минимум 1.
    tolerance : float
        Допуск на шум при проверке стабилизации (ε). Должен быть ≥ 0.
    """
    smoothing_window: int   = 3
    forward_window:  int   = 5
    tolerance:       float = 0.015

    def __post_init__(self) -> None:
        if self.smoothing_window < 1:
            raise ValueError(f"smoothing_window должен быть ≥ 1, получен {self.smoothing_window}")
        if self.forward_window < 1:
            raise ValueError(f"forward_window должен быть ≥ 1, получен {self.forward_window}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance должен быть ≥ 0, получен {self.tolerance}")


# ---------------------------------------------------------------------------
# Вспомогательные функции низкого уровня
# ---------------------------------------------------------------------------

def _smooth_series(ts: pd.Series, window: int) -> pd.Series:
    """
    Возвращает центрированное скользящее среднее ряда.

    Параметры
    ---------
    ts : pd.Series
        Временной ряд с числовым индексом.
    window : int
        Ширина окна (≥ 1).

    Возвращает
    ----------
    pd.Series
        Сглаженный ряд той же длины (min_periods=1 сохраняет крайние значения).
    """
    return ts.rolling(window=window, center=True, min_periods=1).mean()


def _validate_dataframe(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    caller: str = "",
) -> None:
    """
    Проверяет наличие обязательных колонок в датафрейме.

    Исключения
    ----------
    TypeError
        Если df не является pd.DataFrame.
    ValueError
        Если df пустой или отсутствуют обязательные колонки.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{caller}: ожидался pd.DataFrame, получен {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{caller}: датафрейм пустой")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{caller}: отсутствуют колонки {missing}")


def _extract_region_ts(
    df_long:     pd.DataFrame,
    region_name: str,
    region_col:  str = _COL_REGION,
    year_col:    str = _COL_YEAR,
    value_col:   str = _COL_TFR,
) -> pd.Series:
    """
    Извлекает временной ряд СКР для одного региона.

    Возвращает
    ----------
    pd.Series
        СКР, индексированный по году, без NaN, отсортированный.

    Исключения
    ----------
    ValueError
        Если регион не найден в данных или ряд пуст после удаления NaN.
    """
    _validate_dataframe(df_long, [region_col, year_col, value_col], caller="_extract_region_ts")

    mask = df_long[region_col] == region_name
    if not mask.any():
        raise ValueError(f"Регион '{region_name}' не найден в колонке '{region_col}'")

    ts = (
        df_long.loc[mask, [year_col, value_col]]
        .dropna()
        .sort_values(year_col)
        .set_index(year_col)[value_col]
    )

    if ts.empty:
        raise ValueError(f"Временной ряд для региона '{region_name}' пуст после удаления NaN")

    return ts


# ---------------------------------------------------------------------------
# Логика поиска точки стабилизации (методология ООН)
# ---------------------------------------------------------------------------

def _find_stabilization_year(
    ts_smooth:      pd.Series,
    forward_window: int,
    tolerance:      float,
) -> int:
    """
    Находит год стабилизации СКР по адаптированной методологии ООН.

    Алгоритм: первый год, после которого все значения следующих
    `forward_window` лет не опускаются ниже `current_tfr - tolerance`.

    Если такой год не найден — fallback на минимум до `_MATERNAL_CAPITAL_YEAR`.

    Параметры
    ---------
    ts_smooth : pd.Series
        Сглаженный ряд СКР, индексированный по году (int).
    forward_window : int
        Количество лет для подтверждения стабилизации.
    tolerance : float
        Допустимое снижение СКР после предполагаемой точки стабилизации.

    Возвращает
    ----------
    int
        Год начала стабилизации.
    """
    years = ts_smooth.index.tolist()
    n = len(years)

    for i, year in enumerate(years):
        # Нужно как минимум forward_window точек впереди
        if i + forward_window >= n:
            break

        future_slice = ts_smooth.iloc[i + 1 : i + forward_window + 1]
        if future_slice.min() >= (ts_smooth.iloc[i] - tolerance):
            return int(year)

    # Fallback: минимум ряда до введения маткапитала
    fallback_ts = (
        ts_smooth.loc[:_MATERNAL_CAPITAL_YEAR]
        if _MATERNAL_CAPITAL_YEAR in ts_smooth.index
        else ts_smooth
    )
    return int(fallback_ts.idxmin())


# ---------------------------------------------------------------------------
# Назначение фаз на уровне одного региона
# ---------------------------------------------------------------------------

def _assign_phases_for_group(
    group:  pd.DataFrame,
    config: PhaseConfig,
    year_col: str = _COL_YEAR,
    value_col: str = _COL_TFR,
) -> pd.DataFrame:
    """
    Назначает фазы демографического перехода для одного региона.

    Фаза 2 — до года стабилизации включительно.
    Фаза 3 — после года стабилизации.

    Возвращает
    ----------
    pd.DataFrame
        Копия входной группы с добавленной колонкой 'Phase'.
    """
    ts = group.set_index(year_col)[value_col].sort_index()
    ts_smooth = _smooth_series(ts, config.smoothing_window)

    stabilization_year = _find_stabilization_year(
        ts_smooth=ts_smooth,
        forward_window=config.forward_window,
        tolerance=config.tolerance,
    )

    result = group.copy()
    result[_COL_PHASE] = np.where(result[year_col] <= stabilization_year, 2, 3)
    return result


# ---------------------------------------------------------------------------
# Публичный API: назначение фаз
# ---------------------------------------------------------------------------

def process_demographic_phases(
    df_long:    pd.DataFrame,
    config:     PhaseConfig = PhaseConfig(),
    region_col: str = _COL_REGION,
    year_col:   str = _COL_YEAR,
    value_col:  str = _COL_TFR,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Назначает фазы демографического перехода для всех регионов в датафрейме.

    Параметры
    ---------
    df_long : pd.DataFrame
        Панельный датафрейм в длинном формате.
    config : PhaseConfig
        Конфигурация алгоритма.
    region_col, year_col, value_col : str
        Имена соответствующих колонок.

    Возвращает
    ----------
    df_phases : pd.DataFrame
        Входной датафрейм с добавленной колонкой 'Phase'.
    transition_dict : Dict[str, int]
        Словарь {регион: год перехода в Фазу 3}.
    """
    _validate_dataframe(df_long, [region_col, year_col, value_col], caller="process_demographic_phases")

    df_phases = (
        df_long
        .groupby(region_col, sort=False)
        .apply(
            lambda g: _assign_phases_for_group(
                g,
                config,
                year_col=year_col,
                value_col=value_col,
            ),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    transition_dict: Dict[str, int] = (
        df_phases[df_phases[_COL_PHASE] == 3]
        .groupby(region_col)[year_col]
        .min()
        .to_dict()
    )

    return df_phases, transition_dict


# ---------------------------------------------------------------------------
# Публичный API: сенситив-анализ
# ---------------------------------------------------------------------------

def run_phase_sensitivity_analysis(
    df_long:        pd.DataFrame,
    region_name:    str,
    region_col:     str           = _COL_REGION,
    year_col:       str           = _COL_YEAR,
    value_col:      str           = _COL_TFR,
    smoothing_window: int         = 3,
    forward_windows: Iterable[int]   = (3, 4, 5, 6),
    tolerances:     Iterable[float]  = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040),
) -> Tuple[pd.DataFrame, Dict[float, Dict[int, int]]]:
    """
    Сенситив-анализ даты перехода в Фазу 3 по сетке параметров (forward_window × tolerance).

    Сглаживание вычисляется один раз перед перебором сетки — это экономит ресурсы.

    Параметры
    ---------
    df_long : pd.DataFrame
        Панельный датафрейм в длинном формате.
    region_name : str
        Название региона.
    smoothing_window : int
        Фиксированное окно сглаживания (w).
    forward_windows : Iterable[int]
        Перебираемые значения окна подтверждения (k).
    tolerances : Iterable[float]
        Перебираемые значения допуска (ε).

    Возвращает
    ----------
    sensitivity_df : pd.DataFrame
        Таблица чувствительности: строки — tolerance, столбцы — forward_window.
    sensitivity_dict : Dict[float, Dict[int, int]]
        То же в виде вложенного словаря.
    """
    ts = _extract_region_ts(
        df_long=df_long,
        region_name=region_name,
        region_col=region_col,
        year_col=year_col,
        value_col=value_col,
    )

    # Сглаживаем один раз перед перебором сетки
    ts_smooth = _smooth_series(ts, smoothing_window)

    # Перебор сетки параметров
    tol_list = list(tolerances)
    fw_list  = list(forward_windows)

    data = {
        tol: {
            fw: _find_stabilization_year(ts_smooth, fw, tol)
            for fw in fw_list
        }
        for tol in tol_list
    }

    sensitivity_df = (
        pd.DataFrame.from_dict(data, orient="index")
        .rename_axis("tolerance")
        .rename_axis("forward_window", axis="columns")
        .sort_index()
        .sort_index(axis=1)
    )

    # Безопасная конвертация в словарь (NaN пропускаем)
    sensitivity_dict: Dict[float, Dict[int, int]] = {
        float(tol): {
            int(fw): int(year)
            for fw, year in row.items()
            if pd.notna(year)
        }
        for tol, row in sensitivity_df.iterrows()
    }

    return sensitivity_df, sensitivity_dict


# ---------------------------------------------------------------------------
# Визуализация: тепловая карта сенситив-анализа
# ---------------------------------------------------------------------------

def plot_sensitivity_heatmap(
    sensitivity_dict: Dict[float, Dict[int, int]],
) -> plt.Figure:
    """
    Рисует дискретную тепловую карту результатов сенситив-анализа.

    Параметры
    ---------
    sensitivity_dict : Dict[float, Dict[int, int]]
        Словарь {tolerance: {forward_window: year}}.

    Возвращает
    ----------
    plt.Figure
    """
    df = pd.DataFrame.from_dict(sensitivity_dict, orient="index").T
    df.columns.name = "Допуск (tolerance)"
    df.index.name   = "Окно (лет)"

    unique_years = sorted(df.stack().unique())
    n_colors     = max(len(unique_years), 1)
    base_cmap    = plt.get_cmap("RdYlBu_r", n_colors)
    cmap         = mcolors.ListedColormap([base_cmap(i) for i in range(n_colors)])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap=cmap,
        square=True,
        linewidths=1,
        linecolor="white",
        cbar=False,
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Визуализация: исходный и сглаженный СКР
# ---------------------------------------------------------------------------

def plot_smoothed_tfr(
    df_long:     pd.DataFrame,
    region_name: str,
    window:      int           = 3,
    start_year:  Optional[int] = None,
    end_year:    Optional[int] = None,
    year_col:    str           = _COL_YEAR,
    value_col:   str           = _COL_TFR,
    region_col:  str           = _COL_REGION,
) -> plt.Figure:
    """
    График СКР: исходный ряд и его сглаженная версия.

    Параметры
    ---------
    df_long : pd.DataFrame
        Панельный датафрейм.
    region_name : str
        Название региона.
    window : int
        Окно сглаживания.
    start_year, end_year : int, optional
        Границы временного диапазона (включительно). None — без ограничений.

    Возвращает
    ----------
    plt.Figure
    """
    ts = _extract_region_ts(df_long, region_name, region_col, year_col, value_col)

    # Срез временного диапазона (используем is not None, чтобы не ломаться на year=0)
    if start_year is not None:
        ts = ts.loc[ts.index >= start_year]
    if end_year is not None:
        ts = ts.loc[ts.index <= end_year]

    if ts.empty:
        raise ValueError(
            f"Пустой ряд после среза [{start_year}, {end_year}] "
            f"для региона '{region_name}'"
        )

    ts_smooth = _smooth_series(ts, window)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(
        ts.index, ts.values,
        color="#A9A9A9", alpha=0.5,
        label="Исходный СКР",
        linewidth=1.5, linestyle="--",
    )
    ax.plot(
        ts_smooth.index, ts_smooth.values,
        color="#1E88E5",
        label=f"Сглаживание (w={window})",
        linewidth=2.5,
    )

    ax.set_title(f"Анализ макро-тренда СКР: {region_name}", fontsize=13, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    sns.despine()
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Визуализация: как распределены назначенные фазы перехода
# ---------------------------------------------------------------------------

def _prepare_transition_counts(
    transition_dict: Dict[str, int], 
    max_year: int = 2005) -> pd.Series:
    """
    Подфункция: Агрегирует данные о переходах из словаря, отсекая методологические выбросы.

    Параметры:
    ----------
    transition_dict : Dict[str, int]
        Словарь, где ключ — название региона, значение — год начала Фазы III.
    max_year : int
        Год отсечения. Наблюдения, зафиксированные позже этого года, исключаются
        из визуализации (по умолчанию 2003, так как это граница основного распределения).

    Возвращает:
    -----------
    pd.Series
        Отфильтрованный частотный ряд (индекс — год, значения — количество регионов).
    """
    # 1. Извлекаем годы и конвертируем их в pandas Series
    years_series = pd.Series(list(transition_dict.values()))
    
    # 2. Фильтруем ряд, оставляя только те наблюдения, которые не превышают max_year
    filtered_years = years_series[years_series <= max_year]
    
    # 3. Подсчитываем частоту и сортируем индекс по хронологии
    counts_per_year = filtered_years.value_counts().sort_index()
    
    return counts_per_year

def _draw_transition_bars(counts: pd.Series, ax: Axes, bar_width: float = 0.6) -> None:
    """
    Подфункция: Отрисовывает столбчатую диаграмму, искусственно увеличивая
    расстояние между столбцами за счет категориальной оси.

    Параметры:
    ----------
    counts : pd.Series
        Отфильтрованные частотные данные.
    ax : matplotlib.axes.Axes
        Объект оси matplotlib.
    bar_width : float
        Ширина столбцов. Значение < 1 (по умолчанию 0.6) создает больше пустого 
        пространства ("воздуха") между столбцами.

    Возвращает:
    -----------
    None
    """
    # Создаем равномерную сетку координат X (0, 1, 2, 3...) по количеству уникальных лет.
    # Это гарантирует одинаковое широкое расстояние между всеми подписями.
    x_positions = range(len(counts))
    
    # Строим диаграмму, явно задавая ширину столбцов (width)
    bars = ax.bar(x_positions, counts.values, width=bar_width, 
                  color='teal', edgecolor='black', alpha=0.8)
    
    # Добавляем текстовые метки (количество) точно над каждым столбцом
    ax.bar_label(bars, padding=3, fontsize=11, color='black', fontweight='bold')
    
    # Настраиваем текстовые элементы
    ax.set_title(f'Распределение года перехода регионов РФ в Фазу III (до {counts.index.max()} г.)', 
                 fontsize=14, pad=15)
    ax.set_xlabel('Год стабилизации тренда СКР', fontsize=12)
    ax.set_ylabel('Количество субъектов РФ', fontsize=12)
    
    # Привязываем реальные года (индекс Series) к созданной равномерной сетке координат
    ax.set_xticks(x_positions)
    ax.set_xticklabels(counts.index, fontsize=11)
    
    # Эстетическая настройка сетки и рамок
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_transition_distribution(
    transition_dict: Dict[str, int],
    max_year:        int             = 2005,
    bar_width:       float           = 0.6,
    figsize:         Tuple[int, int] = (12, 6),
    note_text:       Optional[str]   = None,
) -> plt.Figure:
    """
    Строит гистограмму распределения регионов по годам перехода в Фазу III.

    Параметры
    ----------
    transition_dict : Dict[str, int]
        Словарь {регион: год первого перехода в Фазу 3}.
    max_year : int, default 2005
        Верхняя граница фильтрации: регионы с переходом после этого года
        исключаются из визуализации как выбросы.
    figsize : Tuple[int, int], default (12, 6)
        Размер полотна в дюймах (ширина, высота).
    note_text : str or None, default None
        Текст сноски под графиком. Если None — формируется автоматически
        на основе max_year.

    Возвращает
    ----------
    plt.Figure
        Объект фигуры matplotlib (plt.show() на стороне вызывающего кода).
    """
    # Фильтруем регионы-выбросы и считаем распределение по годам
    counts = _prepare_transition_counts(transition_dict, max_year=max_year)

    fig, ax = plt.subplots(figsize=figsize)

    # Отрисовка столбчатой диаграммы через вспомогательную функцию
    _draw_transition_bars(counts, ax=ax, bar_width=bar_width)

    # Формируем сноску: пользовательский текст или автоматический по max_year
    _footnote = (
        note_text
        if note_text is not None
        else (
            f"*Примечание: регионы с годом перехода после {max_year} г. "
            "исключены из графика как статистические выбросы."
        )
    )
    fig.text(
        x=0.1, y=0.02,
        s=_footnote,
        ha="left", fontsize=9, style="italic", color="grey",
    )

    # Нижний отступ rect=[0, 0.04, ...] резервирует место для сноски
    plt.tight_layout(rect=[0, 0.04, 1, 1])