"""
=============================================================================
  CRUDE OIL STRATEGIC ANALYSIS — SCHWARTZ & SMITH TWO-FACTOR MODEL
  Kalman Filter + MLE Calibration + Monte Carlo + Stress Test
=============================================================================
  Auteur : Ines TAPOAYI
    Date   : Février 2026
  Modèle : Schwartz & Smith (2000) — "Short-Term Variations and Long-Term
           Dynamics in Commodity Prices", Management Science 46(7).
=============================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = "Db_CL.xlsx"
OUTPUT_DIR  = "outputs"
PILIERS_JOURS = np.array([20, 40, 60])
PILIERS_ANS   = PILIERS_JOURS / 365.25
DT          = 1 / 252
N_SIMS      = 500
HORIZON     = 252          # 1 an de trading
SHOCK_MAG   = 0.30         # +30 % choc pétrolier
VOL_MULT    = 1.5          # ×1.5 volatilité en stress

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
#  1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[1/8] Chargement : {filepath}")
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Décodage des symboles  (format : CLG00, CLH18, …)
    years, months = [], []
    for sym in df["Symbol"]:
        code = str(sym).strip()[-3:]
        m_code, y_code = code[0], code[1:]
        if m_code not in MONTH_CODES:
            years.append(np.nan); months.append(np.nan)
            continue
        years.append(2000 + int(y_code))
        months.append(MONTH_CODES[m_code])

    df["Maturity"] = pd.to_datetime(
        {"year": years, "month": months, "day": 1}, errors="coerce"
    )
    df["Mat_jours"] = (df["Maturity"] - df["Date"]).dt.days
    df["Symbol_clean"] = df["Symbol"].astype(str).str.strip().str[-5:]

    nan_pct = df["Maturity"].isna().mean()
    print(f"    → {len(df):,} lignes | NaN maturité : {nan_pct:.1%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. CONSTRUCTION DES PILIERS FIXES (M20 / M40 / M60) PAR INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────
def build_piliers(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/8] Construction des piliers fixes (M20 / M40 / M60)…")
    df = df.dropna(subset=["Mat_jours", "Close"]).copy()
    df = df[df["Close"] > 0]

    resultats = []
    for i in range(len(df)):
        window = df.iloc[max(0, i - 2): min(len(df), i + 3)]
        if len(window) < 2:
            continue

        x_win = window["Mat_jours"].values.astype(float)
        y_win = np.log(window["Close"].values.astype(float))

        # Suppression des doublons d'abscisses
        _, unique_idx = np.unique(x_win, return_index=True)
        x_win, y_win = x_win[unique_idx], y_win[unique_idx]

        if len(x_win) < 2:
            continue

        f = interp1d(x_win, y_win, kind="linear", fill_value="extrapolate")
        piliers_vals = f(PILIERS_JOURS)

        # Garde-fou : clamp entre log(5$) et log(200$)
        piliers_vals = np.clip(piliers_vals, np.log(5), np.log(200))
        resultats.append([df.iloc[i]["Date"]] + list(piliers_vals))

    df_out = pd.DataFrame(
        resultats, columns=["Date", "M20", "M40", "M60"]
    ).set_index("Date")
    print(f"    → {len(df_out):,} points | Fenêtre : {df_out.index.min().date()} → {df_out.index.max().date()}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
#  3. FILTRE DE KALMAN — SCHWARTZ & SMITH
# ─────────────────────────────────────────────────────────────────────────────
def schwartz_smith_kalman(params, prices_mat, taus, dt):
    """
    Filtre de Kalman vectorisé pour le modèle 2-facteurs Schwartz & Smith.
    H et A sont pré-calculés hors boucle pour maximiser la vitesse.
    """
    kappa, s_chi, s_xi, rho, mu_xi, l_chi, l_xi, meas_err = params

    T = len(prices_mat)
    N = len(taus)
    taus = np.asarray(taus)

    # ── Matrices constantes (calculées une seule fois) ────────────────────
    G = np.array([[np.exp(-kappa * dt), 0.0],
                  [0.0,                  1.0]])
    C = np.array([0.0, (mu_xi - 0.5 * s_xi ** 2) * dt])

    var_chi  = (s_chi ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
    var_xi   = s_xi ** 2 * dt
    cov_term = rho * s_chi * s_xi / kappa * (1 - np.exp(-kappa * dt))
    W = np.array([[var_chi, cov_term],
                  [cov_term, var_xi]])
    W = (W + W.T) / 2
    eig_min = np.linalg.eigvalsh(W).min()
    if eig_min < 0:
        W += (-eig_min + 1e-9) * np.eye(2)

    R = np.eye(N) * max(meas_err, 1e-6)

    # ── H et A pré-calculés (vectorisés sur taus) ─────────────────────────
    H = np.column_stack([np.exp(-kappa * taus), np.ones(N)])   
    A = (-(l_chi / kappa) * (1 - np.exp(-kappa * taus))
         + (mu_xi - l_xi - 0.5 * s_xi ** 2) * taus
         + (s_chi ** 2 / (4 * kappa)) * (1 - np.exp(-2 * kappa * taus))
         + (s_xi ** 2 / 2) * taus
         + (rho * s_chi * s_xi / kappa) * (1 - np.exp(-kappa * taus)))  

    HT      = H.T          
    S_base  = H @ np.zeros((2, 2)) @ HT + R   

    # ── Initialisation ────────────────────────────────────────────────────
    x = np.array([0.0, prices_mat[0, 0]])
    P = np.eye(2) * 0.5

    states  = np.empty((T, 2))
    log_lik = 0.0
    GT      = G.T

    for t in range(T):
        # Prédiction
        x_pred = G @ x + C
        P_pred = G @ P @ GT + W

        # Innovation
        innov = prices_mat[t] - (H @ x_pred + A)
        S     = H @ P_pred @ HT + R

        # Log-vraisemblance
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return np.empty((T, 2)), -1e15
        S_inv    = np.linalg.inv(S)
        log_lik -= 0.5 * (N * np.log(2 * np.pi) + logdet + innov @ S_inv @ innov)

        # Mise à jour
        K = P_pred @ HT @ S_inv
        x = x_pred + K @ innov
        P = (np.eye(2) - K @ H) @ P_pred
        states[t] = x

    return states, log_lik


# ─────────────────────────────────────────────────────────────────────────────
#  4. FONCTION DE LOG-VRAISEMBLANCE (MAXIMISATION MLE)
# ─────────────────────────────────────────────────────────────────────────────
def neg_log_likelihood(params, prices_mat, taus, dt):
    """Minimiser l'opposé de la log-vraisemblance = maximiser la LV."""
    kappa, s_chi = params[0], params[1]
    # Contraintes de stabilité numérique
    if kappa < 1.5 or kappa > 4.0 or s_chi > 0.6:
        return 1e15
    try:
        _, ll = schwartz_smith_kalman(params, prices_mat, taus, dt)
        return -ll if np.isfinite(ll) else 1e15
    except Exception:
        return 1e15


# ─────────────────────────────────────────────────────────────────────────────
#  5. CALIBRATION MLE
# ─────────────────────────────────────────────────────────────────────────────
def calibrate(prices_mat, taus, dt):
    print("[3/8] Calibration MLE (L-BFGS-B)…")

    
    x0 = [2.5, 0.25, 0.15, 0.30, 0.04, 0.0, 0.0, 0.04]
    bounds = [
        (1.5, 4.0),   # kappa 
        (0.01, 0.6),  # s_chi
        (0.01, 0.4),  # s_xi
        (-0.9, 0.9),  # rho
        (0.00, 0.12), # mu_xi
        (-0.2, 0.2),  # l_chi
        (-0.2, 0.2),  # l_xi
        (0.005, 0.15) # meas_err
    ]

    res = minimize(
        neg_log_likelihood, x0,
        args=(prices_mat, taus, dt),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "disp": False}
    )

    if res.success:
        print(f"    ✓ Convergence atteinte | -LL = {res.fun:.2f}")
    else:
        print(f"    ⚠  Convergence partielle : {res.message}")

    params = res.x
    labels = ["κ (mean-reversion)", "σ_χ (vol CT)", "σ_ξ (vol LT)",
              "ρ (corrélation)", "μ_ξ (drift LT)", "λ_χ", "λ_ξ", "σ_ε (mesure)"]
    print("\n    ┌─ PARAMÈTRES CALIBRÉS ──────────────────────────")
    for l, v in zip(labels, params):
        print(f"    │  {l:<26} {v:+.4f}")
    print("    └────────────────────────────────────────────────\n")
    return params


# ─────────────────────────────────────────────────────────────────────────────
#  6. EXTRACTION DES ÉTATS ET MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def extract_states(params, prices_mat, taus, dt):
    print("[4/8] Extraction des états (filtre de Kalman final)…")
    states, ll = schwartz_smith_kalman(params, prices_mat, taus, dt)
    states = np.array(states)

    spot_raw   = np.exp(states[:, 0] + states[:, 1])
    spot_smooth = pd.Series(spot_raw).rolling(5, center=True, min_periods=1).mean().values
    long_term  = np.exp(states[:, 1])

    kappa        = params[0]
    half_life_y  = np.log(2) / kappa
    half_life_d  = half_life_y * 365.25
    half_life_m  = half_life_d / 30.44

    print(f"    Demi-vie des chocs : {half_life_d:.1f} jours ({half_life_m:.1f} mois)")
    print(f"    Log-vraisemblance  : {ll:,.1f}")
    return states, spot_smooth, long_term, half_life_d


# ─────────────────────────────────────────────────────────────────────────────
#  7. SIMULATION DE MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo(params, states, horizon, n_sims, dt, start_date):
    print(f"[5/8] Simulation Monte Carlo ({n_sims} trajectoires, {horizon}j)…")
    kappa, s_chi, s_xi = params[0], params[1], params[2]
    mu_xi              = params[4]
    chi_0, xi_0        = states[-1, 0], states[-1, 1]

    paths = np.zeros((horizon, n_sims))
    for s in range(n_sims):
        chi, xi = chi_0, xi_0
        for t in range(horizon):
            dW1, dW2 = np.random.normal(0, np.sqrt(dt), 2)
            chi += -kappa * chi * dt + s_chi * dW1
            xi  += (mu_xi - 0.5 * s_xi ** 2) * dt + s_xi * dW2
            paths[t, s] = np.exp(chi + xi)

    forecast_dates = pd.bdate_range(start=start_date, periods=horizon)
    p10 = np.percentile(paths, 10, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p90 = np.percentile(paths, 90, axis=1)
    return paths, forecast_dates, p10, p50, p90


# ─────────────────────────────────────────────────────────────────────────────
#  8. STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────
def stress_test(params, states, horizon, n_sims, dt, shock=0.30, vol_mult=1.5):
    print("[6/8] Stress test (choc géopolitique +30%)…")
    kappa_s = params[0] * 0.7          # marché plus inerte
    s_chi_s = params[1] * vol_mult     # volatilité amplifiée
    s_xi    = params[2]
    mu_xi   = params[4]
    chi_0   = states[-1, 0] + np.log(1 + shock)
    xi_0    = states[-1, 1]

    stress_paths = np.zeros((horizon, n_sims))
    for s in range(n_sims):
        chi, xi = chi_0, xi_0
        for t in range(horizon):
            dW1, dW2 = np.random.normal(0, np.sqrt(dt), 2)
            chi += -kappa_s * chi * dt + s_chi_s * dW1
            xi  += (mu_xi - 0.5 * s_xi ** 2) * dt + s_xi * dW2
            stress_paths[t, s] = np.exp(chi + xi)

    hl_stress = np.log(2) / kappa_s * 365.25
    print(f"    Demi-vie stress : {hl_stress:.1f} jours")
    return stress_paths, hl_stress


# ─────────────────────────────────────────────────────────────────────────────
#  9. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
EVENTS = {
    "2008-07-03": "PIC SPÉCULATIF",
    "2014-11-27": "CHOC OPEP",
    "2020-04-20": "KRACH COVID",
    "2022-03-08": "CRISE UKRAINE",
}


def fig_candlestick(df):
    print("    → Candlestick…")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_width=[0.2, 0.8])
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="CL",
        increasing_line_color="#22ab94", decreasing_line_color="#f23645",
        increasing_fillcolor="#22ab94", decreasing_fillcolor="#f23645"
    ), row=1, col=1)
    if "Volume" in df.columns:
        colors = ["#22ab94" if r["Close"] >= r["Open"] else "#f23645"
                  for _, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                             marker_color=colors, name="Volume", opacity=0.3),
                      row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        title="<b>ANALYSE DE MARCHÉ — WTI Crude Oil (CL)</b>",
        hovermode="x unified", showlegend=False,
        margin=dict(l=40, r=50, t=80, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", tickprefix="$", side="right")
    return fig


def fig_decomposition(dates, spot_smooth, long_term, prices_mat, params,
                      states, half_life_d):
    print("    → Décomposition structurelle…")
    kappa_opt = params[0]
    basis_pct = ((spot_smooth - long_term) / long_term) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=np.exp(prices_mat[:, 0]),
                             name="Prix Marché", line=dict(color="lightgrey", width=1),
                             opacity=0.4, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=long_term,
                             name="Valeur Fondamentale (ξ)",
                             line=dict(color="#1A3A6B", width=3),
                             hovertemplate="Équilibre: %{y:.2f}$<extra></extra>"))
    fig.add_trace(go.Scatter(x=dates, y=spot_smooth,
                             name="Spot Modélisé (χ + ξ)",
                             line=dict(color="#D32F2F", width=2),
                             customdata=basis_pct,
                             hovertemplate="<b>Spot: %{y:.2f}$</b><br>Basis: %{customdata:.2f}%<extra></extra>"))

    for ds, label in EVENTS.items():
        d = pd.to_datetime(ds)
        if d in dates:
            y_val = spot_smooth[dates.get_loc(d)]
            fig.add_annotation(x=d, y=y_val, text=f"<b>{label}</b>",
                                showarrow=True, arrowhead=2, ax=0, ay=-55,
                                bgcolor="#1A3A6B", font=dict(color="white", size=10),
                                bordercolor="white", borderwidth=1)

    stats_text = (f"<b>MODEL ANALYTICS</b><br>"
                  f"──────────────────<br>"
                  f"κ (mean-reversion): {kappa_opt:.3f}<br>"
                  f"<b>Demi-vie: {half_life_d:.1f} jours</b><br>"
                  f"σ_ξ (vol LT): {params[2]*100:.1f}%<br>"
                  f"χ actuel: {states[-1,0]:.3f}")
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.96,
                       text=stats_text, showarrow=False, align="left",
                       bgcolor="rgba(255,255,255,0.93)", bordercolor="#1A3A6B",
                       borderwidth=2, borderpad=10,
                       font=dict(family="monospace", size=12, color="#1A3A6B"))
    fig.update_layout(
        title=dict(
            text="<b>CRUDE OIL STRUCTURAL DECOMPOSITION — SCHWARTZ & SMITH</b>"
                 "<br><span style='font-size:13px;color:gray;'>"
                 "Two-Factor Kalman-MLE | Équilibre Fondamental vs Dynamique Cyclique</span>",
            x=0.02),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=40, r=60, t=110, b=100),
        xaxis=dict(showgrid=True, gridcolor="#F2F2F2",
                   rangeselector=dict(buttons=[
                       dict(count=1, label="1a", step="year", stepmode="backward"),
                       dict(count=5, label="5a", step="year", stepmode="backward"),
                       dict(step="all", label="Tout")])),
        yaxis=dict(title="USD / Barrel", side="right", showgrid=True,
                   gridcolor="#F2F2F2", range=[0, 165])
    )
    return fig


def fig_forecast(dates, spot_smooth, long_term, forecast_dates,
                 p10, p50, p90, half_life_d):
    print("    → Fan chart prévision 12 mois…")
    hist_dates = dates[-500:]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist_dates, y=spot_smooth[-500:],
                             name="Historique Spot",
                             line=dict(color="#D32F2F", width=2)))
    fig.add_trace(go.Scatter(x=hist_dates, y=long_term[-500:],
                             name="Équilibre Historique",
                             line=dict(color="#1A3A6B", width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(p90) + list(p10[::-1]),
        fill="toself", fillcolor="rgba(26,58,107,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Intervalle 80%", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(x=forecast_dates, y=p50,
                             name="Projection Médiane (12m)",
                             line=dict(color="black", width=3),
                             hovertemplate="<b>Prévision: %{y:.2f}$</b><extra></extra>"))
    fig.add_vline(x=dates[-1], line_width=2, line_dash="dash", line_color="grey")
    fig.add_annotation(x=dates[-1], y=155,
                       text="DÉBUT PRÉVISION →",
                       showarrow=False, xanchor="left", xshift=10,
                       font=dict(size=12, color="grey"))
    fig.update_layout(
        title="<b>SCÉNARIOS DE PRIX À 12 MOIS</b>"
              "<br><span style='font-size:13px;color:gray;'>"
              f"Monte Carlo ({N_SIMS} sim.) | Demi-vie: {half_life_d:.1f}j</span>",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=60, t=100, b=100),
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0",
                   rangeslider=dict(visible=True, thickness=0.06),
                   rangeselector=dict(buttons=[
                       dict(count=6, label="6m", step="month", stepmode="backward"),
                       dict(count=1, label="1a", step="year", stepmode="backward"),
                       dict(step="all", label="Tout")])),
        yaxis=dict(title="USD / Baril", side="right",
                   showgrid=True, gridcolor="#F0F0F0", range=[0, 165])
    )
    return fig


def fig_risk(forecast_dates, p10, p50, p90, simulated_paths, half_life_d):
    print("    → Profil de risque + distribution…")
    final_prices = simulated_paths[-1, :]
    var_95   = np.percentile(final_prices, 5)
    cvar_95  = final_prices[final_prices <= var_95].mean()

    hl_idx = int(min(half_life_d, len(forecast_dates) - 1))
    hl_date = forecast_dates[hl_idx]
    hl_val  = p50[hl_idx]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32],
                        subplot_titles=("Fan Chart & Convergence", "Distribution Finale"),
                        horizontal_spacing=0.06)
    # Fan chart
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(p90) + list(p10[::-1]),
        fill="toself", fillcolor="rgba(26,58,107,0.12)",
        line=dict(color="rgba(255,255,255,0)"), name="IC 80%"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=p50,
                             line=dict(color="#1A3A6B", width=3), name="Médiane"),
                  row=1, col=1)
    fig.add_annotation(x=hl_date, y=hl_val,
                       text=f"DEMI-VIE<br>{hl_idx}j",
                       showarrow=True, arrowhead=2, arrowcolor="#D32F2F",
                       ax=45, ay=-45, bgcolor="white", bordercolor="#D32F2F",
                       row=1, col=1)
    # Histogram
    fig.add_trace(go.Histogram(x=final_prices, nbinsx=35,
                               marker_color="#1A3A6B", opacity=0.65,
                               name="Distribution Finale"), row=1, col=2)
    fig.add_vline(x=var_95, line_width=2, line_dash="dash",
                  line_color="#D32F2F", row=1, col=2)

    stats = (f"<b>MARKET RISK REPORT</b><br>"
             f"VaR 95% : {var_95:.2f} $<br>"
             f"CVaR (ES) : {cvar_95:.2f} $<br>"
             f"P(prix > VaR) : {(final_prices > var_95).mean()*100:.1f}%")
    fig.add_annotation(xref="paper", yref="paper", x=0.97, y=0.97,
                       text=stats, showarrow=False, align="left",
                       bgcolor="rgba(255,255,255,0.9)",
                       bordercolor="black", borderwidth=1,
                       font=dict(family="monospace", size=11))
    fig.update_layout(title="<b>ANALYSE STOCHASTIQUE & PROFIL DE RISQUE À 12 MOIS</b>",
                      template="plotly_white", hovermode="x unified", height=580)
    return fig


def fig_stress_comparison(forecast_dates, p50, stress_paths, hl_stress):
    print("    → Comparaison stress test…")
    stress_p50 = np.percentile(stress_paths, 50, axis=1)
    stress_p10 = np.percentile(stress_paths, 10, axis=1)
    stress_p90 = np.percentile(stress_paths, 90, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_dates, y=p50,
                             name="Scénario Normal (BAU)",
                             line=dict(color="#1A3A6B", width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(stress_p90) + list(stress_p10[::-1]),
        fill="toself", fillcolor="rgba(211,47,47,0.08)",
        line=dict(color="rgba(255,255,255,0)"), name="IC Stress 80%"
    ))
    fig.add_trace(go.Scatter(x=forecast_dates, y=stress_p50,
                             name="Scénario Stress (+30% / Vol×1.5)",
                             line=dict(color="#D32F2F", width=3)))

    hl_idx_s = int(min(hl_stress, len(forecast_dates) - 1))
    fig.add_annotation(x=forecast_dates[hl_idx_s], y=stress_p50[hl_idx_s],
                       text=f"RÉSILIENCE CRISE<br>Demi-vie: {hl_stress:.0f}j",
                       showarrow=True, arrowhead=2, arrowcolor="#D32F2F",
                       ax=55, ay=-55, bgcolor="white", bordercolor="#D32F2F")
    fig.update_layout(
        title="<b>STRESS TEST — IMPACT D'UN CHOC PÉTROLIER MAJEUR</b>"
              "<br><span style='font-size:13px;color:gray;'>"
              "Comparaison vitesse de retour à l'équilibre fondamental</span>",
        template="plotly_white", hovermode="x unified",
        yaxis=dict(title="USD / Baril", side="right", range=[0, 180]),
        legend=dict(orientation="h", y=-0.18, xanchor="center", x=0.5)
    )
    return fig


def fig_backtest(dates, prices_mat, spot_smooth):
    print("    → Backtest précision modèle…")
    residuals = np.exp(prices_mat[:, 0]) - spot_smooth
    rmse = np.sqrt(np.mean(residuals ** 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=np.exp(prices_mat[:, 0]),
                             name="Prix Réel", line=dict(color="rgba(0,0,0,0.25)", width=1)))
    fig.add_trace(go.Scatter(x=dates, y=spot_smooth,
                             name="Estimation Modèle",
                             line=dict(color="#1A3A6B", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=residuals, name="Erreur (Résiduels)",
                             line=dict(color="rgba(211,47,47,0.5)", width=1),
                             yaxis="y2"))
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.07,
                       text=f"<b>RMSE: {rmse:.2f} $ / baril</b>",
                       showarrow=False, bgcolor="white",
                       font=dict(size=13, color="#1A3A6B"))
    fig.update_layout(
        title="<b>BACKTEST — PRÉCISION DU MODÈLE SCHWARTZ & SMITH</b>",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(title="Temps"),
        yaxis=dict(title="Prix USD / Baril"),
        yaxis2=dict(title="Erreur (USD)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.18)
    )
    return fig


def save_all_figs(figs_dict: dict):
    print("\n[7/8] Sauvegarde des graphiques HTML…")
    for name, fig in figs_dict.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.html")
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"    ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  10. EXPORT POWER BI / EXCEL
# ─────────────────────────────────────────────────────────────────────────────
def export_powerbi(dates, prices_mat, spot_smooth, long_term,
                   forecast_dates, p10, p50, p90,
                   stress_paths, params, half_life_d):
    print("[8/8] Export CSV (Power BI / Excel)…")

    # Série temporelle
    df_hist = pd.DataFrame({
        "Date":        dates,
        "Prix_Obs":    np.exp(prices_mat[:, 0]),
        "Equilibre_LT": long_term,
        "Spot_Modele": spot_smooth,
        "Scenario":    "Historique"
    })
    df_bau = pd.DataFrame({
        "Date":        forecast_dates,
        "Spot_Modele": p50,
        "IC_Bas":      p10,
        "IC_Haut":     p90,
        "Scenario":    "Normal (BAU)"
    })
    df_stress = pd.DataFrame({
        "Date":        forecast_dates,
        "Spot_Modele": np.percentile(stress_paths, 50, axis=1),
        "Scenario":    "Stress Test"
    })
    df_ts = pd.concat([df_hist, df_bau, df_stress], axis=0)
    ts_path = os.path.join(OUTPUT_DIR, "PowerBI_Oil_TimeSeries.csv")
    df_ts.to_csv(ts_path, index=False, sep=";", decimal=",")
    print(f"    ✓ {ts_path}")

    # Métriques KPI
    final_prices = None   # recalculé depuis df_bau
    var_95 = np.percentile(p50, 5)
    cvar_95 = p50[p50 <= var_95].mean() if (p50 <= var_95).any() else p50.min()
    prob_conv = ((p50 >= long_term[-1] * 0.9) & (p50 <= long_term[-1] * 1.1)).mean() * 100
    hl_stress = np.log(2) / (params[0] * 0.7) * 365.25

    metrics = [
        {"Indicateur": "Mean Reversion (κ)",    "Valeur": params[0],     "Unite": "vitesse"},
        {"Indicateur": "Demi-vie (Normal)",      "Valeur": half_life_d,   "Unite": "jours"},
        {"Indicateur": "Demi-vie (Stress)",      "Valeur": hl_stress,     "Unite": "jours"},
        {"Indicateur": "σ_χ (vol court terme)",  "Valeur": params[1]*100, "Unite": "%"},
        {"Indicateur": "σ_ξ (vol long terme)",   "Valeur": params[2]*100, "Unite": "%"},
        {"Indicateur": "VaR 95%",                "Valeur": var_95,        "Unite": "USD"},
        {"Indicateur": "CVaR / Expected Shortfall","Valeur": cvar_95,     "Unite": "USD"},
        {"Indicateur": "Prob. Convergence (±10%)","Valeur": prob_conv/100,"Unite": "%"},
    ]
    df_m = pd.DataFrame(metrics)
    m_path = os.path.join(OUTPUT_DIR, "PowerBI_Oil_Metrics.csv")
    df_m.to_csv(m_path, index=False, sep=";", decimal=",")
    print(f"    ✓ {m_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CRUDE OIL ANALYSIS — SCHWARTZ & SMITH TWO-FACTOR MODEL")
    print("=" * 70 + "\n")

    # 1. Données
    df_base = load_and_clean(DATA_FILE)

    # 2. Piliers
    df_piliers = build_piliers(df_base)
    prices_mat = df_piliers.values
    dates      = df_piliers.index

    # 3–4. Calibration + états
    params          = calibrate(prices_mat, PILIERS_ANS, DT)
    states, spot_smooth, long_term, half_life_d = extract_states(
        params, prices_mat, PILIERS_ANS, DT
    )

    # 5. Monte Carlo
    sim_paths, forecast_dates, p10, p50, p90 = monte_carlo(
        params, states, HORIZON, N_SIMS, DT, start_date=dates[-1]
    )

    # 6. Stress test
    stress_paths, hl_stress = stress_test(
        params, states, HORIZON, N_SIMS, DT, SHOCK_MAG, VOL_MULT
    )

    # 7. Graphiques
    print("[7/8] Génération des visualisations…")
    figs = {
        "01_candlestick":       fig_candlestick(df_base.set_index("Date")),
        "02_decomposition":     fig_decomposition(dates, spot_smooth, long_term,
                                                   prices_mat, params, states, half_life_d),
        "03_forecast_12m":      fig_forecast(dates, spot_smooth, long_term,
                                              forecast_dates, p10, p50, p90, half_life_d),
        "04_risk_profile":      fig_risk(forecast_dates, p10, p50, p90,
                                          sim_paths, half_life_d),
        "05_stress_test":       fig_stress_comparison(forecast_dates, p50,
                                                       stress_paths, hl_stress),
        "06_backtest":          fig_backtest(dates, prices_mat, spot_smooth),
    }
    save_all_figs(figs)

    # 8. Export
    export_powerbi(dates, prices_mat, spot_smooth, long_term,
                   forecast_dates, p10, p50, p90, stress_paths, params, half_life_d)

    print("\n✅  ANALYSE TERMINÉE.")
    print(f"    Fichiers HTML dans : ./{OUTPUT_DIR}/")
    print(f"    Fichiers CSV dans  : ./{OUTPUT_DIR}/")

    # Ouverture automatique du dashboard principal
    import webbrowser
    main_html = os.path.abspath(os.path.join(OUTPUT_DIR, "02_decomposition.html"))
    webbrowser.open(f"file://{main_html}")


if __name__ == "__main__":
    main()
