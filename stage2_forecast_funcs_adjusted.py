"""
funcs.py — Иерархическая AR(1)-модель с региональными ориентирами.

Мотивация
---------
Базовая модель Alkema предполагает единый центр возврата μ_RU для всех регионов.
На российской региональной панели это порождает систематическое смещение:
завышение прогнозов для низкофертильных регионов и занижение для высокофертильных.

Решение — региональная адаптация при сохранении структуры AR(1):

    μ_r = μ_RU + α_r                              (региональный ориентир)
    f_{r,t+1} = f_{r,t} + ρ(μ_r − f_{r,t}) + ε_t (динамика Фазы III)

Здесь ρ остаётся глобальным (единая скорость возврата), а гетерогенность
переносится в долгосрочный уровень μ_r.

Пайплайн
--------
1. estimate_hierarchical_params()  — двухэтапная оценка ρ и {α_r}
2. simulate_regional_forecast()    — МК-симуляция для одного региона
3. run_hierarchical_backtest()     — бэктест по всем регионам
4. evaluate_hierarchical_calibration() — Bias Test + Calibration Test
5. plot_regional_attractors()      — визуализация α_r
6. plot_regional_forecast()        — веерная диаграмма прогноза
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Имена колонок (согласованы с demographic_phases.py и demographic_forecast.py)
# ---------------------------------------------------------------------------
_COL_REGION = "Регион"
_COL_YEAR   = "Year"
_COL_TFR    = "TFR"

# Внутренние расчётные колонки
_COL_TFR_DIFF    = "_tfr_diff"     # f_{t+1} − f_t
_COL_TFR_GAP_GLO = "_tfr_gap_glo"  # μ_RU − f_t  (глобальный gap)
_COL_RESID_RAW   = "_resid_raw"    # остаток глобального OLS
_COL_RESID_INNOV = "_resid_innov"  # инновационный остаток после вычета ρ·α_r

# Колонки выходного прогноза
_COL_MEDIAN = "Median"
_COL_LOWER  = "Lower_95"
_COL_UPPER  = "Upper_95"
_COL_MU_R   = "mu_r"               # региональный ориентир в прогнозном DF


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchicalConfig:
    """
    Параметры иерархической AR(1)-модели.

    Атрибуты
    --------
    federal_target : float
        Федеральный ориентир μ_RU. По умолчанию 1.6 (прогноз Минтруда РФ).
    max_transition_year : int
        Регионы, перешедшие в Фазу III после этого года, исключаются из
        обучающей выборки как выбросы.
    min_phase3_obs : int
        Минимальное число наблюдений Фазы III для оценки α_r региона.
        Регионы с меньшим количеством точек получают α_r = 0 (fallback).
    n_sims : int
        Число симуляций Монте-Карло (≥ 1).
    ci_lower_pct : float
        Нижний перцентиль доверительного интервала.
    ci_upper_pct : float
        Верхний перцентиль доверительного интервала.
    """
    federal_target:      float = 1.6
    max_transition_year: int   = 2005
    min_phase3_obs:      int   = 5
    n_sims:              int   = 10_000
    ci_lower_pct:        float = 2.5
    ci_upper_pct:        float = 97.5
    seed:                int   = 42

    def __post_init__(self) -> None:
        if self.federal_target <= 0:
            raise ValueError(f"federal_target должен быть > 0, получен {self.federal_target}")
        if self.min_phase3_obs < 2:
            raise ValueError(f"min_phase3_obs должен быть ≥ 2, получен {self.min_phase3_obs}")
        if self.n_sims < 1:
            raise ValueError(f"n_sims должен быть ≥ 1, получен {self.n_sims}")
        if not (0 < self.ci_lower_pct < self.ci_upper_pct < 100):
            raise ValueError(
                f"Некорректные границы ДИ: "
                f"ci_lower_pct={self.ci_lower_pct}, ci_upper_pct={self.ci_upper_pct}"
            )


@dataclass(frozen=True)
class BacktestConfig:
    """
    Параметры процедуры бэктестинга.

    Атрибуты
    --------
    cutoff_year : int
        Граница обучения: модель видит только данные ≤ cutoff_year.
    n_forecast_years : int
        Горизонт прогноза от cutoff_year (≥ 1).
    """
    cutoff_year:      int = 2020
    n_forecast_years: int = 5

    def __post_init__(self) -> None:
        if self.n_forecast_years < 1:
            raise ValueError(f"n_forecast_years должен быть ≥ 1, получен {self.n_forecast_years}")


# ---------------------------------------------------------------------------
# Расчётные структуры данных
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionalAttractors:
    """
    Региональные ориентиры, оценённые из данных Фазы III.

    Атрибуты
    --------
    alpha : Dict[str, float]
        Региональные поправки α_r = μ_r − μ_RU.
    mu : Dict[str, float]
        Региональные ориентиры μ_r = μ_RU + α_r.
    federal_target : float
        Федеральный ориентир μ_RU.
    fallback_regions : List[str]
        Регионы, получившие α_r = 0 из-за нехватки данных.

    Примечания
    ----------
    mu[r] = federal_target + alpha[r] выполняется для всех r.
    """
    alpha:            Dict[str, float]
    mu:               Dict[str, float]
    federal_target:   float
    fallback_regions: List[str]

    def get_mu(self, region: str) -> float:
        """Возвращает μ_r для региона, с fallback на federal_target."""
        return self.mu.get(region, self.federal_target)


@dataclass(frozen=True)
class HierarchicalParams:
    """
    Полный набор оценённых параметров иерархической модели.

    Атрибуты
    --------
    rho : float
        Глобальный коэффициент скорости возврата (единый для всех регионов).
    s : float
        Стандартное отклонение инновационных ошибок ε_t.
    attractors : RegionalAttractors
        Региональные ориентиры μ_r.
    n_obs : int
        Число наблюдений, использованных при оценке.
    """
    rho:        float
    s:          float
    attractors: RegionalAttractors
    n_obs:      int


@dataclass(frozen=True)
class CalibrationResult:
    """
    Сводные метрики калибровки модели.

    Атрибуты
    --------
    bias_rate : float
        Доля случаев, когда факт > медианы. Идеал — 0.50.
    error_rate : float
        Доля случаев, когда факт вышел за пределы ДИ. Идеал — 0.05.
    n_obs : int
        Число наблюдений в тесте.
    detail : pd.DataFrame
        Детальный датафрейм с флагами is_above_median, is_outside_ci.
    """
    bias_rate:  float
    error_rate: float
    n_obs:      int
    detail:     pd.DataFrame


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------

def _validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    caller: str = "",
) -> None:
    """Проверяет тип, непустоту и наличие обязательных колонок."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{caller}: ожидался pd.DataFrame, получен {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{caller}: датафрейм пустой")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{caller}: отсутствуют колонки {missing}")


def _get_start_tfr(
    tfr_index: pd.Series,
    region: str,
    year: int,
) -> float:
    """
    Безопасно извлекает СКР региона в заданном году.

    Исключения
    ----------
    ValueError
        Если данных нет или значение NaN.
    """
    try:
        value = tfr_index.loc[(region, year)]
    except KeyError:
        raise ValueError(
            f"Для региона '{region}' отсутствуют данные за {year} год."
        )
    if pd.isna(value):
        raise ValueError(
            f"СКР региона '{region}' за {year} год равен NaN."
        )
    return float(value)


# ---------------------------------------------------------------------------
# Шаг 1: Подготовка обучающей выборки Фазы III
# ---------------------------------------------------------------------------

def _prepare_phase3_dataset(
    df: pd.DataFrame,
    transition_dict: Dict[str, int],
    config: HierarchicalConfig,
) -> pd.DataFrame:
    """
    Формирует обучающую выборку: только наблюдения Фазы III для «стабильного
    ядра» регионов (перешедших ≤ max_transition_year).

    Рассчитываемые колонки
    ----------------------
    _COL_TFR_DIFF    : f_{t+1} − f_t
    _COL_TFR_GAP_GLO : μ_RU − f_t  (gap относительно федерального ориентира)

    Исключения
    ----------
    ValueError
        Если после фильтрации не остаётся регионов или наблюдений.
    """
    _validate_dataframe(df, [_COL_REGION, _COL_YEAR, _COL_TFR], caller="_prepare_phase3_dataset")

    valid_regions = {
        reg for reg, yr in transition_dict.items()
        if yr <= config.max_transition_year
    }
    if not valid_regions:
        raise ValueError(
            f"Нет регионов, перешедших в Фазу III до {config.max_transition_year}"
        )

    def _region_features(group: pd.DataFrame) -> pd.DataFrame:
        region    = group[_COL_REGION].iloc[0]
        trans_yr  = transition_dict[region]
        phase3    = group[group[_COL_YEAR] >= trans_yr].sort_values(_COL_YEAR).copy()

        phase3[_COL_TFR_DIFF]    = phase3[_COL_TFR].shift(-1) - phase3[_COL_TFR]
        phase3[_COL_TFR_GAP_GLO] = config.federal_target - phase3[_COL_TFR]

        # Последняя строка региона не имеет f_{t+1} → удаляем
        return phase3.dropna(subset=[_COL_TFR_DIFF])

    result = (
        df[df[_COL_REGION].isin(valid_regions)]
        .groupby(_COL_REGION, sort=False)
        .apply(_region_features, include_groups=True)
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError("Обучающая выборка пуста после подготовки.")

    return result


# ---------------------------------------------------------------------------
# Шаг 2: Двухэтапная оценка ρ и {α_r}
# ---------------------------------------------------------------------------

def _estimate_global_rho(df_phase3: pd.DataFrame) -> Tuple[float, pd.Series]:
    """
    Этап 1: OLS без константы по всей панели → глобальный ρ.

    Модель: TFR_diff = ρ · (μ_RU − TFR) + ε
    (ε содержит и инновационную ошибку, и систематические региональные сдвиги)

    Возвращает
    ----------
    rho : float
    raw_residuals : pd.Series  (индекс совпадает с df_phase3)
    """
    Y = df_phase3[_COL_TFR_DIFF]
    X = df_phase3[_COL_TFR_GAP_GLO]

    model = sm.OLS(Y, X).fit()
    rho   = float(model.params.iloc[0])

    return rho, model.resid


def _estimate_regional_alphas(
    df_phase3:     pd.DataFrame,
    raw_residuals: pd.Series,
    rho:           float,
    config:        HierarchicalConfig,
) -> RegionalAttractors:
    """
    Этап 2: из остатков глобального OLS извлекаем α_r.

    Вывод:
        TFR_diff_{r,t} = ρ·(μ_RU − f_{r,t}) + ρ·α_r + ε_{r,t}
        ê_{r,t}        = TFR_diff_{r,t} − ρ̂·(μ_RU − f_{r,t})
        E[ê_{r,t}]     = ρ̂·α_r
        → α̂_r          = mean(ê_{r,t}) / ρ̂

    Регионы с числом наблюдений < min_phase3_obs получают α_r = 0 (fallback).

    Параметры
    ---------
    df_phase3 : pd.DataFrame
        Обучающая выборка (содержит _COL_REGION).
    raw_residuals : pd.Series
        Остатки глобального OLS (индекс совпадает с df_phase3).
    rho : float
        Оценённый глобальный коэффициент.
    config : HierarchicalConfig

    Возвращает
    ----------
    RegionalAttractors
    """
    if abs(rho) < 1e-10:
        raise ValueError(
            f"rho ≈ 0 ({rho:.2e}): деление на rho невозможно. "
            "Проверьте данные или спецификацию модели."
        )

    df_work = df_phase3[[_COL_REGION]].copy()
    df_work[_COL_RESID_RAW] = raw_residuals.values

    region_stats = df_work.groupby(_COL_REGION)[_COL_RESID_RAW].agg(["mean", "count"])

    fallback_regions: List[str] = []
    alpha_dict: Dict[str, float] = {}

    for region, row in region_stats.iterrows():
        if row["count"] < config.min_phase3_obs:
            alpha_dict[region] = 0.0
            fallback_regions.append(region)
        else:
            alpha_dict[region] = row["mean"] / rho

    mu_dict = {
        reg: config.federal_target + alpha
        for reg, alpha in alpha_dict.items()
    }

    return RegionalAttractors(
        alpha=alpha_dict,
        mu=mu_dict,
        federal_target=config.federal_target,
        fallback_regions=fallback_regions,
    )


def _estimate_innovation_std(
    df_phase3:  pd.DataFrame,
    raw_residuals: pd.Series,
    rho:        float,
    attractors: RegionalAttractors,
) -> float:
    """
    Оценивает стандартное отклонение инновационной ошибки s.

    После вычета систематического регионального сдвига ρ·α_r
    остаток ε_{r,t} = ê_{r,t} − ρ·α_r должен быть белым шумом.
    s = std(ε) по всей панели.
    """
    df_work = df_phase3[[_COL_REGION]].copy()
    df_work[_COL_RESID_RAW] = raw_residuals.values

    # Вычитаем систематический сдвиг каждого региона
    df_work[_COL_RESID_INNOV] = df_work.apply(
        lambda row: row[_COL_RESID_RAW] - rho * attractors.alpha.get(row[_COL_REGION], 0.0),
        axis=1,
    )

    n = len(df_work)
    s = float(np.sqrt(np.sum(df_work[_COL_RESID_INNOV] ** 2) / (n - 1)))
    return s


# ---------------------------------------------------------------------------
# Публичный API: оценка параметров
# ---------------------------------------------------------------------------

def estimate_hierarchical_params(
    df: pd.DataFrame,
    transition_dict: Dict[str, int],
    config: HierarchicalConfig = HierarchicalConfig(),
    cutoff_year: Optional[int] = None,
) -> HierarchicalParams:
    """
    Оркестратор двухэтапной оценки иерархической AR(1)-модели.

    Этап 1: Глобальный ρ из OLS по всей панели Фазы III.
    Этап 2: Региональные поправки α_r из систематики остатков.

    Параметры
    ---------
    df : pd.DataFrame
        Панельный датафрейм с колонками [Регион, Year, TFR].
    transition_dict : Dict[str, int]
        Словарь {регион: год перехода в Фазу III}.
    config : HierarchicalConfig
        Конфигурация модели.
    cutoff_year : int, optional
        Если задан, обучение ведётся только на данных ≤ cutoff_year.

    Возвращает
    ----------
    HierarchicalParams
        Глобальный ρ, волатильность s и региональные ориентиры μ_r.
    """
    _validate_dataframe(df, [_COL_REGION, _COL_YEAR, _COL_TFR], caller="estimate_hierarchical_params")

    df_train = df[df[_COL_YEAR] <= cutoff_year].copy() if cutoff_year is not None else df.copy()

    df_phase3 = _prepare_phase3_dataset(df_train, transition_dict, config)

    rho, raw_residuals = _estimate_global_rho(df_phase3)
    attractors         = _estimate_regional_alphas(df_phase3, raw_residuals, rho, config)
    s                  = _estimate_innovation_std(df_phase3, raw_residuals, rho, attractors)

    return HierarchicalParams(
        rho=rho,
        s=s,
        attractors=attractors,
        n_obs=len(df_phase3),
    )


# ---------------------------------------------------------------------------
# Монте-Карло симуляция
# ---------------------------------------------------------------------------

def _simulate_regional_ensemble(
    start_tfr: float,
    mu_r:      float,
    params:    HierarchicalParams,
    config:    HierarchicalConfig,
    n_years:   int,
    rng:       np.random.Generator,
) -> np.ndarray:
    """
    Генерирует ансамбль траекторий для одного региона.

    Уравнение шага:
        f_{t+1} = f_t + ρ·(μ_r − f_t) + ε_t,   ε_t ~ N(0, s²)

    Параметры
    ---------
    start_tfr : float
        Начальное значение СКР.
    mu_r : float
        Региональный ориентир.
    params : HierarchicalParams
        Оценённые параметры.
    config : HierarchicalConfig
        Конфигурация (n_sims).
    n_years : int
        Горизонт прогноза.
    rng : np.random.Generator
        Общий генератор случайных чисел (для воспроизводимости).

    Возвращает
    ----------
    np.ndarray
        Матрица (n_sims × n_years).
    """
    ensemble = np.empty((config.n_sims, n_years))
    current  = np.full(config.n_sims, start_tfr)

    for t in range(n_years):
        errors         = rng.normal(loc=0.0, scale=params.s, size=config.n_sims)
        current        = current + params.rho * (mu_r - current) + errors
        ensemble[:, t] = current

    return ensemble


def _build_forecast_dataframe(
    ensemble:   np.ndarray,
    start_year: int,
    mu_r:       float,
    config:     HierarchicalConfig,
) -> pd.DataFrame:
    """
    Агрегирует ансамбль в таблицу медианы и ДИ.

    Возвращает
    ----------
    pd.DataFrame
        Колонки: Year, Median, Lower_95, Upper_95, mu_r.
    """
    n_years = ensemble.shape[1]
    years   = np.arange(start_year + 1, start_year + 1 + n_years)

    pcts = np.percentile(
        ensemble,
        [50, config.ci_lower_pct, config.ci_upper_pct],
        axis=0,
    )

    return pd.DataFrame({
        _COL_YEAR:   years,
        _COL_MEDIAN: pcts[0],
        _COL_LOWER:  pcts[1],
        _COL_UPPER:  pcts[2],
        _COL_MU_R:   mu_r,
    })


def simulate_regional_forecast(
    region:     str,
    start_tfr:  float,
    start_year: int,
    n_years:    int,
    params:     HierarchicalParams,
    config:     HierarchicalConfig = HierarchicalConfig()
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Стохастическая симуляция прогноза для одного региона.

    Параметры
    ---------
    region : str
        Название региона (для поиска μ_r).
    start_tfr : float
        СКР в базовом году.
    start_year : int
        Базовый год (не входит в прогноз).
    n_years : int
        Горизонт прогноза (≥ 1).
    params : HierarchicalParams
        Оценённые параметры модели.
    config : HierarchicalConfig
        Конфигурация.
    seed : int, optional
        Зерно ГСЧ для воспроизводимости.

    Возвращает
    ----------
    df_forecast : pd.DataFrame
        Медиана и ДИ по каждому году прогноза, плюс колонка mu_r.
    ensemble : np.ndarray
        Сырая матрица (n_sims × n_years).
    """
    if n_years < 1:
        raise ValueError(f"n_years должен быть ≥ 1, получен {n_years}")

    mu_r = params.attractors.get_mu(region)
    rng  = np.random.default_rng(config.seed)

    ensemble    = _simulate_regional_ensemble(start_tfr, mu_r, params, config, n_years, rng)
    df_forecast = _build_forecast_dataframe(ensemble, start_year, mu_r, config)

    return df_forecast, ensemble


# ---------------------------------------------------------------------------
# Бэктестинг
# ---------------------------------------------------------------------------

def run_hierarchical_backtest(
    df_long:         pd.DataFrame,
    transition_dict: Dict[str, int],
    hier_config:     HierarchicalConfig = HierarchicalConfig(),
    bt_config:       BacktestConfig     = BacktestConfig()
) -> pd.DataFrame:
    """
    Обучает иерархическую модель на данных ≤ cutoff_year, строит прогноз
    для каждого региона из transition_dict и собирает результаты.

    Параметры
    ---------
    df_long : pd.DataFrame
        Панельный датафрейм.
    transition_dict : Dict[str, int]
        Словарь {регион: год перехода в Фазу III}.
    hier_config : HierarchicalConfig
    bt_config : BacktestConfig
    seed : int, optional
        Зерно ГСЧ (единое для всех регионов для воспроизводимости).

    Возвращает
    ----------
    pd.DataFrame
        Прогнозы для всех регионов с колонкой Регион.
    """
    _validate_dataframe(df_long, [_COL_REGION, _COL_YEAR, _COL_TFR], caller="run_hierarchical_backtest")

    cutoff = bt_config.cutoff_year
    params = estimate_hierarchical_params(
        df_long, transition_dict, config=hier_config, cutoff_year=cutoff
    )

    tfr_index = df_long.set_index([_COL_REGION, _COL_YEAR])[_COL_TFR]
    rng  = np.random.default_rng(hier_config.seed)

    regional_forecasts: List[pd.DataFrame] = []

    for region in transition_dict:
        start_tfr = _get_start_tfr(tfr_index, region, cutoff)
        mu_r      = params.attractors.get_mu(region)

        ensemble    = _simulate_regional_ensemble(
            start_tfr, mu_r, params, hier_config, bt_config.n_forecast_years, rng
        )
        df_forecast = _build_forecast_dataframe(ensemble, cutoff, mu_r, hier_config)
        df_forecast[_COL_REGION] = region
        regional_forecasts.append(df_forecast)

    return pd.concat(regional_forecasts, ignore_index=True)


# ---------------------------------------------------------------------------
# Оценка калибровки
# ---------------------------------------------------------------------------

def evaluate_hierarchical_calibration(
    backtest_df: pd.DataFrame,
    df_long:     pd.DataFrame,
) -> CalibrationResult:
    """
    Bias Test и Calibration Test для иерархической модели.

    Параметры
    ---------
    backtest_df : pd.DataFrame
        Результат run_hierarchical_backtest.
    df_long : pd.DataFrame
        Панель с фактическими значениями СКР.

    Возвращает
    ----------
    CalibrationResult
        Сводные метрики + детальный датафрейм с флагами.
    """
    _validate_dataframe(
        backtest_df,
        [_COL_YEAR, _COL_REGION, _COL_MEDIAN, _COL_LOWER, _COL_UPPER],
        caller="evaluate_hierarchical_calibration",
    )
    _validate_dataframe(df_long, [_COL_YEAR, _COL_REGION, _COL_TFR],
                        caller="evaluate_hierarchical_calibration")

    merged = pd.merge(
        backtest_df,
        df_long[[_COL_YEAR, _COL_REGION, _COL_TFR]],
        on=[_COL_YEAR, _COL_REGION],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "После объединения прогноза с фактом — пустой датафрейм. "
            "Проверьте совпадение годов и регионов."
        )

    merged["is_above_median"] = merged[_COL_TFR] > merged[_COL_MEDIAN]
    merged["is_outside_ci"]   = (
        (merged[_COL_TFR] < merged[_COL_LOWER]) |
        (merged[_COL_TFR] > merged[_COL_UPPER])
    )

    return CalibrationResult(
        bias_rate  = float(merged["is_above_median"].mean()),
        error_rate = float(merged["is_outside_ci"].mean()),
        n_obs      = len(merged),
        detail     = merged,
    )


# ---------------------------------------------------------------------------
# Визуализация 1: Региональные ориентиры (Cleveland dot plot)
# ---------------------------------------------------------------------------

def plot_regional_attractors(
    attractors:   RegionalAttractors,
    top_n:        Optional[int] = None,
    highlight:    Optional[List[str]] = None,
) -> plt.Figure:
    """
    Cleveland dot plot региональных поправок α_r и ориентиров μ_r.

    Регионы сортируются по α_r. Регионы из fallback_regions
    отмечаются другим маркером. Можно выделить конкретные регионы.

    Параметры
    ---------
    attractors : RegionalAttractors
        Оценённые региональные ориентиры.
    top_n : int, optional
        Если задан, показываются только top_n регионов с наибольшими |α_r|.
    highlight : List[str], optional
        Регионы, которые нужно выделить цветом.

    Возвращает
    ----------
    plt.Figure
    """
    df_plot = (
        pd.DataFrame({
            "region": list(attractors.alpha.keys()),
            "alpha":  list(attractors.alpha.values()),
            "mu_r":   list(attractors.mu.values()),
        })
        .sort_values("alpha")
        .reset_index(drop=True)
    )

    if top_n is not None:
        df_plot = (
            df_plot
            .reindex(df_plot["alpha"].abs().nlargest(top_n).index)
            .sort_values("alpha")
            .reset_index(drop=True)
        )

    highlight_set = set(highlight or [])
    fallback_set  = set(attractors.fallback_regions)
    n_regions     = len(df_plot)
    fig_height    = max(4, n_regions * 0.28)

    fig, ax = plt.subplots(figsize=(7, fig_height), dpi=150)

    # Горизонтальные направляющие (lollipop stems)
    for _, row in df_plot.iterrows():
        ax.plot(
            [0, row["alpha"]], [row["region"], row["region"]],
            color="#D0D0D0", linewidth=0.8, zorder=1,
        )

    # Точки
    for _, row in df_plot.iterrows():
        region = row["region"]
        color  = (
            "#E53935" if region in highlight_set else
            "#9E9E9E" if region in fallback_set else
            ("#1E88E5" if row["alpha"] >= 0 else "#FB8C00")
        )
        marker = "D" if region in fallback_set else "o"
        ax.scatter(
            row["alpha"], region,
            color=color, s=40, marker=marker, zorder=3, linewidths=0,
        )

    # Вертикальная нулевая линия — граница федерального ориентира
    ax.axvline(
        0, color="#455A64", linewidth=1.2, linestyle="--",
        label=f"μ_RU = {attractors.federal_target}",
    )

    ax.set_xlabel("Региональная поправка α_r = μ_r − μ_RU", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    sns.despine(left=True)

    # Легенда
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E88E5",
               markersize=7, label="α_r > 0 (выше федерального)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FB8C00",
               markersize=7, label="α_r < 0 (ниже федерального)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#9E9E9E",
               markersize=7, label="Fallback (мало данных)"),
        Line2D([0], [0], linestyle="--", color="#455A64",
               label=f"μ_RU = {attractors.federal_target}"),
    ]
    if highlight_set:
        legend_items.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935",
                   markersize=7, label="Выделенный регион")
        )

    # Легенда вынесена вниз вплотную к графику
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
        frameon=False,
        ncol=5,
    )

    # Небольшой нижний отступ под внешнюю легенду без лишней пустоты
    fig.subplots_adjust(bottom=0.10)

# ---------------------------------------------------------------------------
# Визуализация 2: Веерная диаграмма прогноза региона
# ---------------------------------------------------------------------------

def plot_regional_forecast(
    df_forecast:    pd.DataFrame,
    df_history:     pd.DataFrame,
    region:         str,
    attractors:     RegionalAttractors,
    start_year:     int,
    year_col:       str = _COL_YEAR,
    tfr_col:        str = _COL_TFR,
    region_col:     str = _COL_REGION,
) -> plt.Figure:
    """
    Веерная диаграмма: история СКР + медиана прогноза + 95% ДИ + μ_r.

    Параметры
    ---------
    df_forecast : pd.DataFrame
        Результат simulate_regional_forecast (Median, Lower_95, Upper_95, mu_r).
    df_history : pd.DataFrame
        Исходная панель (для отрисовки исторического ряда).
    region : str
        Название региона.
    attractors : RegionalAttractors
        Региональные ориентиры (для подписи μ_r).
    start_year : int
        Год разделения истории и прогноза.

    Возвращает
    ----------
    plt.Figure
    """
    _validate_dataframe(df_forecast, [_COL_YEAR, _COL_MEDIAN, _COL_LOWER, _COL_UPPER],
                        caller="plot_regional_forecast")
    _validate_dataframe(df_history, [region_col, year_col, tfr_col],
                        caller="plot_regional_forecast")

    hist = (
        df_history[df_history[region_col] == region]
        .sort_values(year_col)
    )
    if hist.empty:
        raise ValueError(f"Нет исторических данных для региона '{region}'")

    mu_r  = attractors.get_mu(region)
    alpha = attractors.alpha.get(region, 0.0)
    mu_ru = attractors.federal_target

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # --- История ---
    ax.plot(
        hist[year_col], hist[tfr_col],
        color="#37474F", linewidth=2, label="Факт СКР",
    )

    # --- Переходная вертикаль ---
    ax.axvline(start_year, color="#90A4AE", linewidth=1, linestyle="--", alpha=0.8)

    # --- Прогноз: ДИ и медиана ---
    ax.fill_between(
        df_forecast[_COL_YEAR],
        df_forecast[_COL_LOWER],
        df_forecast[_COL_UPPER],
        color="#1E88E5", alpha=0.15,
        label=f"95% ДИ",
    )
    ax.plot(
        df_forecast[_COL_YEAR], df_forecast[_COL_MEDIAN],
        color="#1E88E5", linewidth=2.5, label="Медиана прогноза",
    )

    # --- Региональный ориентир μ_r ---
    all_years = list(hist[year_col]) + list(df_forecast[_COL_YEAR])
    ax.axhline(
        mu_r, color="#E53935", linewidth=1.2, linestyle=":",
        label=f"μ_r = {mu_r:.3f}  (α_r = {alpha:+.3f})",
    )

    # --- Федеральный ориентир μ_RU (пунктир) ---
    ax.axhline(
        mu_ru, color="#FB8C00", linewidth=1, linestyle="--", alpha=0.7,
        label=f"μ_RU = {mu_ru} (федеральный)",
    )

    # Аннотация разницы μ_r − μ_RU в правой части
    x_ann = df_forecast[_COL_YEAR].max()
    mid_y = (mu_r + mu_ru) / 2
    ax.annotate(
        f"Δ = {alpha:+.3f}",
        xy=(x_ann, mid_y),
        xytext=(4, 0), textcoords="offset points",
        fontsize=8, color="#E53935", va="center",
    )

    ax.set_title(f"Иерархический прогноз СКР: {region}", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("СКР")
    
    # Легенда вынесена вниз под график
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15), 
        fontsize=8, 
        frameon=False,
        ncol=5
    )
    
    ax.grid(True, linestyle=":", alpha=0.4)
    sns.despine()
    
    # rect=[0, 0.05, 1, 1] оставляет место снизу для вынесенной легенды
    plt.tight_layout(rect=[0, 0.05, 1, 1])

# ЕЩЕ БЭКТЕСТЫ

def _run_single_hierarchical_backtest(
    df_long: pd.DataFrame,
    transition_dict: dict,
    h_config: HierarchicalConfig,
    cutoff_year: int,
    n_forecast_years: int,
) -> dict:
    """
    Выполняет один прогон иерархического бэктеста для заданного периода обучения.

    Параметры
    ----------
    df_long : pd.DataFrame
        Панельный датафрейм с историческими значениями СКР.
    transition_dict : dict
        Словарь {регион: год перехода в Фазу III}.
    h_config : HierarchicalConfig
        Конфигурация иерархической модели.
    cutoff_year : int
        Последний год обучающей выборки.
    n_forecast_years : int
        Горизонт бэктеста в годах.

    Возвращает
    ----------
    dict
        Словарь с итоговыми метриками и параметрами модели для данного cutoff.
    """
    # Оцениваем параметры модели только на данных до cutoff_year
    params = estimate_hierarchical_params(
        df=df_long,
        transition_dict=transition_dict,
        config=h_config,
        cutoff_year=cutoff_year,
    )

    # Строим бэктест на том же cutoff_year
    bt_config = BacktestConfig(
        cutoff_year=cutoff_year,
        n_forecast_years=n_forecast_years,
    )
    backtest_results = run_hierarchical_backtest(
        df_long=df_long,
        transition_dict=transition_dict,
        hier_config=h_config,
        bt_config=bt_config,
    )

    # Оцениваем калибровку медианы и доверительного интервала
    calib = evaluate_hierarchical_calibration(
        backtest_df=backtest_results,
        df_long=df_long,
    )

    return {
        "cutoff_year": cutoff_year,
        "n_forecast_years": n_forecast_years,
        "rho": params.rho,
        "s": params.s,
        "n_obs": params.n_obs,
        "bias": calib.bias_rate,
        "error": calib.error_rate,
    }

def run_hierarchical_period_comparison(
    df_long: pd.DataFrame,
    transition_dict: dict,
    federal_target: float,
    cutoff_years: list[int],
    n_forecast_years: int = 5,
) -> pd.DataFrame:
    """
    Сравнивает качество иерархической модели при разных границах обучающего периода.

    Параметры
    ----------
    df_long : pd.DataFrame
        Панельный датафрейм с историческими значениями СКР.
    transition_dict : dict
        Словарь {регион: год перехода в Фазу III}.
    federal_target : float
        Федеральный ориентир μ_RU.
    cutoff_years : list[int]
        Список годов, до которых ограничивается обучение модели.
    n_forecast_years : int, default 5
        Горизонт бэктеста в годах.

    Возвращает
    ----------
    pd.DataFrame
        Сводная таблица по метрикам качества для каждого cutoff_year.
    """
    h_config = HierarchicalConfig(
        federal_target=federal_target,
    )

    results_list = []

    for cutoff_year in cutoff_years:
        result = _run_single_hierarchical_backtest(
            df_long=df_long,
            transition_dict=transition_dict,
            h_config=h_config,
            cutoff_year=cutoff_year,
            n_forecast_years=n_forecast_years,
        )
        results_list.append(result)

    df_results = (
        pd.DataFrame(results_list)
        .sort_values("cutoff_year")
        .reset_index(drop=True)
    )

    return df_results