import numpy as np
import pandas as pd
from typing import List, Optional
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Глобальный генератор случайных чисел для воспроизводимости результатов
_rng = np.random.default_rng(seed=42)

def set_seed(seed: int = 42) -> None:
    """Устанавливает глобальный seed для воспроизводимости симуляций."""
    global _rng
    _rng = np.random.default_rng(seed=seed)

def generate_policy_multipliers(
    beta_mean: float, 
    beta_se: float, 
    rmc_coverage: np.ndarray, 
    use_log_scaling: bool = True, 
    n_sims: int = 10_000
) -> np.ndarray:
    """
    Генерирует стохастическую матрицу мультипликаторов эффекта РМК.
    
    Методология: используется байесовский подход с генерацией коэффициента 
    из нормального распределения N(beta_mean, beta_se^2).
    """
    # Логарифмическое масштабирование отражает убывающую предельную полезность
    shocks = np.log1p(rmc_coverage) if use_log_scaling else rmc_coverage
    
    # Генерация коэффициента влияния (Beta) с учётом стандартной ошибки
    beta_samples = _rng.normal(loc=beta_mean, scale=beta_se, size=n_sims).reshape(-1, 1)
    
    # Расчёт мультипликаторов через экспоненциальную связь (для лог-линейных моделей)
    return np.exp(beta_samples @ shocks.reshape(1, -1))

def report_policy_impact(
    baseline_ensemble: np.ndarray,
    policy_multipliers: np.ndarray,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Формирует аналитический отчёт о влиянии РМК на прогноз СКР.

    Параметры
    ----------
    baseline_ensemble : np.ndarray
        Матрица базовых прогнозных траекторий СКР размерности
        (n_sims, n_years).
    policy_multipliers : np.ndarray
        Матрица мультипликаторов эффекта РМК размерности
        (n_sims, n_years).
    years : Optional[List[int]], default None
        Список прогнозных лет. Если None, формируется автоматически.

    Возвращает
    ----------
    pd.DataFrame
        Таблица с медианными значениями базового прогноза, итогового
        прогноза и прироста по каждому году.
    """
    if years is None:
        years = list(range(2026, 2026 + baseline_ensemble.shape[1]))

    med_base = np.median(baseline_ensemble, axis=0)
    med_mult = np.median(policy_multipliers, axis=0)
    med_policy = np.median(baseline_ensemble * policy_multipliers, axis=0)

    data = []
    for i, year in enumerate(years):
        effect_pct = (med_mult[i] - 1.0) * 100
        abs_gain = med_policy[i] - med_base[i]

        data.append({
            "Year": year,
            "Baseline_TFR": med_base[i],
            "Policy_Effect_Pct": effect_pct,
            "Absolute_Gain": abs_gain,
            "Resulting_TFR": med_policy[i],
        })

    df_report = pd.DataFrame(data)

    print("=" * 68)
    print(" АНАЛИЗ ЭФФЕКТА РМК НА ПРОГНОЗ СКР")
    print("=" * 68)

    for _, row in df_report.iterrows():
        print(
            f"Год {int(row['Year'])}: "
            f"{row['Baseline_TFR']:.3f} → {row['Resulting_TFR']:.3f} "
            f"(+{row['Absolute_Gain']:.3f}; +{row['Policy_Effect_Pct']:.2f}%)"
        )

    print("=" * 68)

    return df_report

import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_fan_chart(baseline_df: pd.DataFrame, policy_ensemble: np.ndarray,
                       last_hist_year: int = 2025, last_hist_val: float = 1.287, 
                       target: float = 1.6):
    
    # 1. Подготовка данных (добавляем историческую точку для "сшивки")
    years = [int(col.replace('year_', '')) for col in baseline_df.columns]
    proj_years = [last_hist_year] + years
    
    def get_fan_data(data, hist_val):
        # Добавляем историческую точку в начало каждой симуляции
        extended = np.hstack([np.full((data.shape[0], 1), hist_val), data])
        return {
            'p95': np.percentile(extended, 97.5, axis=0),
            'p80': np.percentile(extended, 90, axis=0),
            'p50': np.percentile(extended, 50, axis=0),
            'p20': np.percentile(extended, 10, axis=0),
            'p05': np.percentile(extended, 2.5, axis=0)
        }

    base = get_fan_data(baseline_df.values, last_hist_val)
    policy = get_fan_data(policy_ensemble, last_hist_val)

    fig = go.Figure()

    # --- Цветовая палитра ---
    c_base = 'rgba(150, 150, 150, ' # Серый
    c_poly = 'rgba(30, 136, 229, '  # Насыщенный синий

    def add_fan_traces(fig, data, color_prefix, name_suffix, show_legend=True):
        # Внешнее облако (95%)
        fig.add_trace(go.Scatter(x=proj_years, y=data['p95'], mode='lines', line_width=0, showlegend=False))
        fig.add_trace(go.Scatter(x=proj_years, y=data['p05'], mode='lines', line_width=0, 
                                 fill='tonexty', fillcolor=f'{color_prefix}0.1)', name=f'95% ИИ {name_suffix}'))
        # Внутреннее облако (50%)
        fig.add_trace(go.Scatter(x=proj_years, y=data['p80'], mode='lines', line_width=0, showlegend=False))
        fig.add_trace(go.Scatter(x=proj_years, y=data['p20'], mode='lines', line_width=0, 
                                 fill='tonexty', fillcolor=f'{color_prefix}0.25)', name=f'50% ИИ {name_suffix}'))
        # Медиана
        line_style = dict(color=f'{color_prefix}1)', width=3) if "РМК" in name_suffix else dict(color='gray', dash='dash', width=2)
        fig.add_trace(go.Scatter(x=proj_years, y=data['p50'], mode='lines', line=line_style, name=f'Медиана {name_suffix}'))

    # Рисуем
    add_fan_traces(fig, base, c_base, '(База)')
    add_fan_traces(fig, policy, c_poly, '(РМК)')

    # --- Таргет (сделаем его зоной, а не просто линией) ---
    fig.add_hrect(y0=target, y1=target+0.005, fillcolor="red", opacity=0.3, line_width=0)
    fig.add_annotation(x=proj_years[1], y=target, text=f"Цель Минтруда: {target}", 
                       showarrow=False, yshift=10, font=dict(color="red", size=12), xanchor="left")

    # --- Аннотация эффекта в конце периода ---
    final_idx = -1
    delta = policy['p50'][final_idx] - base['p50'][final_idx]
    fig.add_annotation(
        x=proj_years[final_idx], y=policy['p50'][final_idx],
        text=f"Эффект РМК: +{delta:.2f} п.п.",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        bgcolor="white", bordercolor=c_poly+"1)", borderwidth=1
    )

    fig.update_layout(
        title=dict(text="<b>Прогноз СКР: Инерционный сценарий vs Региональный маткапитал</b>", font_size=20),
        xaxis=dict(title="Год", gridcolor='white', showline=True, linecolor='black', dtick=1),
        yaxis=dict(title="СКР (детей на женщину)", gridcolor='rgba(0,0,0,0.05)', range=[1.1, 1.7], zeroline=False),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=100, b=60)
    )

    fig.show()

def plot_joyplot_with_delta(baseline_df, policy_ensemble, title: str = 'РМК', target=1.6):
    # Преобразование данных
    base_arr = baseline_df.values if isinstance(baseline_df, pd.DataFrame) else baseline_df
    poly_arr = policy_ensemble.values if isinstance(policy_ensemble, pd.DataFrame) else policy_ensemble
    
    years_labels = [col.replace('year_', '') for col in baseline_df.columns]
    num_years = baseline_df.shape[1]
    
    fig, axes = plt.subplots(num_years, 1, figsize=(11, 8), sharex=True)
    plt.subplots_adjust(hspace=-0.6)
    
    c_base, c_poly, c_success = '#E0E0E0', '#3B82F6', '#10B981'
    x = np.linspace(1.0, 1.8, 300)
    
    for i in range(num_years):
        ax = axes[i]
        z = num_years - i 
        
        base_data = base_arr[:, i]
        poly_data = poly_arr[:, i]
        
        kde_b = gaussian_kde(base_data)(x)
        kde_p = gaussian_kde(poly_data)(x)
        
        scale = 0.8 / max(max(kde_b), max(kde_p))
        y_b, y_p = kde_b * scale, kde_p * scale

        # Отрисовка Базы и Политики
        ax.fill_between(x, y_b, color=c_base, alpha=0.7, zorder=z)
        ax.plot(x, y_b, color='#9CA3AF', lw=1, alpha=0.5, zorder=z)
        ax.fill_between(x, y_p, color=c_poly, alpha=0.6, zorder=z+0.1)
        ax.plot(x, y_p, color='white', lw=1.5, zorder=z+0.2)

        # Зеленая зона успеха
        mask = x >= target
        ax.fill_between(x[mask], y_p[mask], color=c_success, alpha=0.8, zorder=z+0.3)
        ax.axvline(target, color='red', lw=1.2, ls='--', alpha=0.4, zorder=10)
        
        # --- НОВЫЙ РАСЧЕТ ---
        prob_base = np.mean(base_data >= target) * 100
        prob_poly = np.mean(poly_data >= target) * 100
        delta = prob_poly - prob_base
        
        # Вывод текста: Общий % и в скобках (+ прибавка)
        label_text = f'{prob_poly:.1f}%'
        if delta > 0:
            label_text += f' (+{delta:.1f}%)'
            
        ax.text(target + 0.06, 0.1, label_text, 
                fontsize=9, fontweight='bold', color='#065F46', 
                ha='left', zorder=20, transform=ax.get_xaxis_transform())
        
        # Косметика осей
        ax.set_yticks([])
        ax.set_facecolor('none')
        ax.spines[['left', 'top', 'right', 'bottom']].set_visible(False)
        ax.text(-0.02, 0.1, years_labels[i], transform=ax.transAxes, 
                fontweight='bold', fontsize=12, color='#374151', ha='right')

    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].spines['bottom'].set_color('#D1D5DB')
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    axes[-1].set_xlabel("Суммарный коэффициент рождаемости")
    
    plt.show()