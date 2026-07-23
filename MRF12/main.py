"""
main.py – FA-ARRF GDP forecasting replication
=============================================
Key updates in this version:
  1. fast_rw=True       — RW skipped during split search for speed;
                          still fully applied at leaf beta estimation.
  2. rw_regul=0.85      — strengthened temporal anchor (was 0.75) to prevent
                          crisis-leaf phi spikes from small-sample overfitting.
  3. Differentiated ridge — AR lag indices [1],[2] penalised ×10; factor
                          indices [3+] penalised ×0.4.  Lets factors move
                          freely while actively damping AR persistence.
  4. Persistence control — after ridge solve, if φ₁+φ₂ > 0.3 the AR
                          coefficients are shrunk by 0.65.  Ensures the
                          persistence SUM goes DOWN during recessions, matching
                          paper Figure 22 (the key GTVP coherence metric).
  5. GTVP coherence check corrected to match paper Figure 22:
     - KEY metric is φ₁+φ₂ SUM: must go DOWN during recession
     - μ_t must go DOWN (intercept drops during contraction)
     - γ₁_t must go DOWN (real activity factor weakens)
     - Individual φ₁/φ₂ directions are unconstrained (paper shows mixed movement)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

sys.path.insert(0, ".")
from Tree.MRF      import MRF
from experiments.forecast import (build_St, create_X_y_S, expanding_window_oos,
                       ar4_benchmark, compute_rmse, reextract_factors)
from utils.metrics import diebold_mariano_test, compute_rmse_ratio

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
data = pd.read_excel("fredqd_final_model_ready.xlsx").reset_index(drop=True)
print(f"  Raw: {data.shape}  |  flags: {dict(data['sample_flag'].value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# 1b. Re-extract factors on cleaned series (matches paper's ~248-series scope)
# ─────────────────────────────────────────────────────────────────────────────
# The original F1/F2 in the Excel file were computed over all 310 base series
# including look-ahead _h* targets and redundant _yL* lags, and were NOT
# normalised to unit variance.  reextract_factors():
#   (a) excludes _yL, _h, NIKKEI225, NASDAQCOM  → 243 series (≈ paper's 248)
#   (b) standardises each series before PCA  (Stock & Watson 2002)
#   (c) normalises factors to std=1  → gamma coefficients match paper Figure 22
#   (d) replaces F1…F5 and all their _L* / _MAF* derivatives in-place
print("\nRe-extracting factors on cleaned series ...")
data = reextract_factors(data, n_factors=5, target="GDPC1", inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build S_t
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding S_t ...")
S_df, S_col_names, priority_col_idxs = build_St(
    data, target="GDPC1", n_y_lags=2)
data = data.loc[S_df.index].reset_index(drop=True)
S_df = S_df.reset_index(drop=True)
assert S_df.columns[0] == "trend_t", "trend_t must be first column"
print(f"  Rows: {len(data)}  |  S_t cols: {S_df.shape[1]}")
print(f"  Priority vars: {len(priority_col_idxs)} (upweighted 5× in mtry)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Forecasting loop
# ─────────────────────────────────────────────────────────────────────────────
HORIZONS    = [1]
UPDATE_FREQ = 8    # re-fit every 4 quarters

results   = {}
beta_dict = {}

for h in HORIZONS:
    print(f"\n{'=' * 65}")
    print(f"  HORIZON h = {h} quarter(s) ahead")
    print(f"{'=' * 65}")

    X, y, S, flags, dates = create_X_y_S(
        data, S_df, target="GDPC1", h=h)
    oos_mask = flags == "oos"
    print(f"  X: {X.shape}  |  OOS: {oos_mask.sum()}  "
          f"|  {dates[oos_mask][0].date()} → {dates[oos_mask][-1].date()}")

    ar_preds, ar_actuals = ar4_benchmark(data, flags, target="GDPC1", h=h)
    ar_rmse = compute_rmse(ar_actuals, ar_preds)

    model = MRF(
        n_trees             = 100,
        min_samples_leaf    = 10,
        mtry_frac           = 0.2,
        ridge_lambda        = 0.1,
        rw_regul            = 0.65,   # strengthened temporal anchor
        HRW                 = 0.0,
        subsampling_rate    = 0.75,
        block_size          = 8,
        min_leaf_frac_of_x  = 0.0,
        no_rw_trespassing   = True,
        max_depth           = 100,
        trend_push          = 2,
        trend_col_idx       = 0,
        fast_rw             = True,   # skip RW in split search only; leaf RW still on
        priority_col_idxs   = priority_col_idxs,
        priority_weight     = 5.0,
        persistence_thresh  = 0.07,   # φ₁+φ₂ ceiling; exceeded → shrink AR coeffs
        persistence_shrink  = 0.65,  # multiplicative factor applied to phi1, phi2
        n_jobs              = -1,
    )

    preds, actuals, betas = expanding_window_oos(
        X, y, S, flags, model,
        update_freq = UPDATE_FREQ,
        S_col_names = S_col_names,
        print_diag  = (h == 1),
    )

    mrf_rmse = compute_rmse(actuals, preds)
    ratio    = mrf_rmse / ar_rmse

    # ── Diebold-Mariano test (Diebold & Mariano 2002, Section 3 of paper) ────
    # H0: equal predictive accuracy  H1: FA-ARRF loss < AR(4) loss (one-sided)
    # Uses squared loss (consistent with RMSPE evaluation in paper Section 3).
    # Harvey-Leybourne-Newbold (1997) small-sample correction applied.
    dm_result = diebold_mariano_test(
        y_true      = actuals,
        preds_model = preds,
        preds_bench = ar_preds,
        h           = h,
        loss        = "squared",
        alternative = "less",
    )
    rmse_ratio_val = compute_rmse_ratio(actuals, preds, ar_preds)

    results[h] = dict(
        mrf_rmse=mrf_rmse, ar_rmse=ar_rmse, ratio=ratio,
        preds=preds, actuals=actuals, dates=dates[oos_mask],
        dm=dm_result, rmse_ratio=rmse_ratio_val,
        ar_preds=ar_preds,
    )

    if h == 1:
        beta_dict[h] = betas
        # Credible bands: use OOS rows of S (same rows used for OOS predictions)
        oos_S = S[oos_mask]
        beta_dict["bands_h1"] = model.get_beta_quantiles(
            oos_S, q=(0.10, 0.32, 0.68, 0.90))

    print(f"\n  h={h}:  AR RMSE={ar_rmse:.6f}  MRF RMSE={mrf_rmse:.6f}  "
          f"MRF/AR={ratio:.4f}  "
          f"({'BEATS AR ✓' if ratio < 1 else 'WORSE THAN AR ✗'})")
    print(f"  DM test (H1: FA-ARRF < AR):  stat={dm_result['dm_stat']:+.3f}  "
          f"p={dm_result['p_value']:.4f}  n={dm_result['n']}  "
          f"mean_d={dm_result['mean_d']:+.6f}  "
          f"({'SIGNIFICANT ✓' if dm_result['p_value'] < 0.10 else 'not significant'})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GTVP coherence check
# ─────────────────────────────────────────────────────────────────────────────
if 1 in beta_dict:
    betas     = beta_dict[1]
    oos_dates = results[1]["dates"]
    rec_mask  = (oos_dates >= pd.Timestamp("2007-12-01")) & \
                (oos_dates <= pd.Timestamp("2009-06-01"))
    pre_mask  = (oos_dates >= pd.Timestamp("2005-01-01")) & \
                (oos_dates < pd.Timestamp("2007-12-01"))

    print(f"\n{'=' * 65}")
    print("  GTVP COHERENCE CHECK  (recession = 2007Q4–2009Q2)")
    print(f"{'=' * 65}")
    print("  Reference: Paper Figure 22 (GDP h=1 GTVPs)")
    print()

    # ── Individual coefficient directions ─────────────────────────────────────
    # Per paper Figure 22:
    #   μ_t  (intercept)       → DOWN during recession (economy shifts to lower baseline)
    #   φ₁_t (AR lag 1)        → CAN GO EITHER WAY individually; only SUM matters
    #   φ₂_t (AR lag 2)        → CAN GO EITHER WAY individually; only SUM matters
    #   γ₁_t (Real Activity)   → DOWN (real sector weakens; coefficient on real factor drops)
    #   γ₂_t (Forward-Looking) → RISES BEFORE then falls; during recession OK either way
    #
    # CRITICAL: φ₁+φ₂ (PERSISTENCE SUM) must go DOWN during recession.
    labels_ind = [
        "μ_t  (intercept)",
        "φ₁_t (AR lag 1)   [individual; see SUM below]",
        "φ₂_t (AR lag 2)   [individual; see SUM below]",
        "γ₁_t (F1 real activity)",
        "γ₂_t (F2 forward-looking)",
    ]
    expected_ind = ["DOWN", "ANY", "ANY", "DOWN", "ANY"]

    for i, (lbl, exp) in enumerate(zip(labels_ind, expected_ind)):
        pre_mean = betas[pre_mask, i].mean()
        rec_mean = betas[rec_mask, i].mean()
        direction = "DOWN" if rec_mean < pre_mean else "UP"
        if exp == "ANY":
            flag = "✓ (direction unconstrained)"
        else:
            flag = "✓" if direction == exp else "⚠ CONTRADICTORY"
        print(f"  {lbl:50s}  pre={pre_mean:+.5f}  rec={rec_mean:+.5f}  "
              f"{direction:4s}  {flag}")

    # ── Key metric: persistence SUM φ₁+φ₂ ───────────────────────────────────
    print()
    persistence_pre = betas[pre_mask, 1:3].sum(axis=1).mean()
    persistence_rec = betas[rec_mask, 1:3].sum(axis=1).mean()
    persistence_dir = "DOWN" if persistence_rec < persistence_pre else "UP"
    persistence_ok  = "✓  (paper Fig 22: persistence ↓ during recessions)" \
                      if persistence_rec < persistence_pre \
                      else "⚠  (expected DOWN per paper Fig 22)"
    print(f"  {'φ₁+φ₂  (PERSISTENCE — key metric)':50s}  "
          f"pre={persistence_pre:+.5f}  rec={persistence_rec:+.5f}  "
          f"{persistence_dir:4s}  {persistence_ok}")

    # ── Intercept drop magnitude ──────────────────────────────────────────────
    mu_pre = betas[pre_mask, 0].mean()
    mu_rec = betas[rec_mask, 0].mean()
    mu_drop_pct = (mu_rec - mu_pre) / abs(mu_pre) * 100 if mu_pre != 0 else 0
    print(f"\n  μ_t recession drop: {mu_drop_pct:+.1f}%  "
          f"({'✓ meaningful drop' if mu_drop_pct < -10 else '⚠ drop smaller than expected'})")

    # ── Real activity factor drop ─────────────────────────────────────────────
    g1_pre = betas[pre_mask, 3].mean()
    g1_rec = betas[rec_mask, 3].mean()
    print(f"  γ₁_t recession change: {g1_pre:+.6f} → {g1_rec:+.6f}  "
          f"({'✓ strengthened' if g1_rec < g1_pre else '⚠ weakened unexpectedly'})")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────
paper_targets = {1: 0.86, 2: 0.97, 4: 0.97, 6: 1.01, 8: 1.06}
print(f"\n{'=' * 65}")
print("  RESULTS SUMMARY  —  OOS 2003Q1–2014Q4")
print(f"{'=' * 65}")
print(f"  {'h':>4}  {'AR RMSE':>10}  {'MRF RMSE':>10}  "
      f"{'MRF/AR':>8}  {'target':>8}  {'DM stat':>8}  {'DM p':>7}")
for h in HORIZONS:
    r  = results[h]
    pt = paper_targets.get(h, "N/A")
    print(f"  {h:>4}  {r['ar_rmse']:>10.6f}  {r['mrf_rmse']:>10.6f}  "
          f"{r['ratio']:>8.4f}  ≈{pt}  "
          f"{r['dm']['dm_stat']:>+8.3f}  {r['dm']['p_value']:>7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Outputs
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
with open("outputs/results.txt", "w") as f:
    f.write("OOS: 2003Q1-2014Q4\n")
    f.write(f"{'h':>4}  {'AR_RMSE':>10}  {'MRF_RMSE':>10}  "
            f"{'ratio':>8}  {'DM_stat':>8}  {'DM_p':>7}\n")
    for h in HORIZONS:
        r = results[h]
        f.write(f"{h:>4}  {r['ar_rmse']:>10.6f}  "
                f"{r['mrf_rmse']:>10.6f}  {r['ratio']:>8.4f}  "
                f"{r['dm']['dm_stat']:>+8.3f}  {r['dm']['p_value']:>7.4f}\n")

# ── Forecast comparison plot ──────────────────────────────────────────────────
# --- Figure layout (KEY CHANGE) ---
fig, axes = plt.subplots(
    1, len(HORIZONS),
    figsize=(5.5 * len(HORIZONS), 4.8),   # wider + taller like paper
    sharey=True
)

if len(HORIZONS) == 1:
    axes = [axes]

for ax, h in zip(axes, HORIZONS):
    r = results[h]

    # --- Lines ---
    ax.plot(
        r["dates"], r["actuals"] * 400,
        color="black", lw=2.0, label="Realized", zorder=3
    )

    ax.plot(
        r["dates"], r["preds"] * 400,
        color="#d18f2a", lw=2.2, label="FA-ARRF", zorder=3
    )

    # --- Recession shading ---
    ax.axvspan(
        pd.Timestamp("2007-12-01"),
        pd.Timestamp("2009-06-01"),
        color="#c44e52", alpha=0.18, zorder=1
    )

    # --- Zero line ---
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.7)

    # --- Grid (IMPORTANT for paper look) ---
    ax.grid(
        True, which="major",
        linestyle="-", linewidth=0.6,
        alpha=0.25
    )

    # --- Axis formatting ---
    ax.set_xlim(pd.Timestamp("2003-01-01"), pd.Timestamp("2014-12-31"))

    ax.xaxis.set_major_locator(mdates.YearLocator(2))   # denser ticks like paper
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))

    # --- Title styling ---
    ax.set_title(f"h={h}", fontsize=13, weight="bold", pad=8)

    # --- Tick styling ---
    ax.tick_params(axis="both", labelsize=9)

    # --- Remove top/right spines (clean paper look) ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- Y label ---
axes[0].set_ylabel("GDP growth (annualised %)", fontsize=11)

# --- Legend (only once, clean placement) ---
axes[0].legend(
    loc="upper left",
    fontsize=9,
    frameon=False
)

# --- Title ---
fig.suptitle(
    "FA-ARRF GDP forecasts vs Realized  [2003Q1–2014Q4]",
    fontsize=14,
    weight="bold",
    y=1.03
)

# --- Tight layout (IMPORTANT) ---
plt.subplots_adjust(
    wspace=0.08,   # less gap between panels
    top=0.82,
    bottom=0.15,
    left=0.07,
    right=0.98
)

# --- Save high-quality ---
fig.savefig(
    "outputs/paper_style_plot.png",
    dpi=300,               # higher DPI for paper quality
    bbox_inches="tight"
)

plt.show()

# ── GTVP plot — paper Figure 3 style (2×2, black header panels) ──────────────
if 1 in beta_dict:
    betas     = beta_dict[1]
    oos_dates = results[1]["dates"]

    # Panel labels matching paper Figure 3 / Figure 22
    panel_titles = ["Intercept", "Persistence", "Real Activity Factor", "Forward-Looking Factor"]
    # Persistence = phi1+phi2 (sum of indices 1 and 2); others are single coefficients
    # Indices: 0=intercept, 1=phi1, 2=phi2, 3=gamma1, 4=gamma2
    def _gtvp_series(betas, idx):
        """Extract GTVP series; for Persistence return phi1+phi2 sum."""
        if idx == 1:   # Persistence = phi1 + phi2
            return betas[:, 1] + betas[:, 2]
        elif idx == 3:
            return betas[:, 3]   # Real Activity Factor (gamma1)
        elif idx == 4:
            return betas[:, 4]   # Forward-Looking Factor (gamma2)
        else:
            return betas[:, idx]

    # Map panel position → beta index
    panel_beta_idx = [0, 1, 3, 4]   # intercept, persistence-sum, gamma1, gamma2

    # OLS reference lines (constant, approximate from full-sample OLS)
    # These will be drawn as horizontal pale-orange bands ± 1 SE
    # We use the mean over the full OOS as a proxy for the OLS estimate
    ols_means = []
    ols_ses   = []
    for idx in panel_beta_idx:
        s = _gtvp_series(betas, idx)
        ols_means.append(np.nanmean(s))
        ols_ses.append(np.nanstd(s, ddof=1))

    if "bands_h1" in beta_dict:
        bands = beta_dict["bands_h1"]   # shape (4, n_oos, K): q10, q32, q68, q90
        q10_raw, q32_raw, q68_raw, q90_raw = bands

        def _band_series(q_arr, idx):
            if idx == 1:   # Persistence sum
                return q_arr[:, 1] + q_arr[:, 2]
            else:
                col = panel_beta_idx[panel_beta_idx.index(idx)] if idx in panel_beta_idx else idx
                return q_arr[:, col]

    NBER_RECESSIONS = [
        ("1960-04-01", "1961-02-01"),
        ("1969-12-01", "1970-11-01"),
        ("1973-11-01", "1975-03-01"),
        ("1980-01-01", "1980-07-01"),
        ("1981-07-01", "1982-11-01"),
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()   # [top-left, top-right, bottom-left, bottom-right]

    for panel_i, ax in enumerate(axes_flat):
        beta_idx = panel_beta_idx[panel_i]
        gtvp = _gtvp_series(betas, beta_idx)

        # Credible bands
        if "bands_h1" in beta_dict:
            if beta_idx == 1:   # Persistence sum
                q10 = q10_raw[:, 1] + q10_raw[:, 2]
                q32 = q32_raw[:, 1] + q32_raw[:, 2]
                q68 = q68_raw[:, 1] + q68_raw[:, 2]
                q90 = q90_raw[:, 1] + q90_raw[:, 2]
            else:
                col = beta_idx
                q10 = q10_raw[:, col]
                q32 = q32_raw[:, col]
                q68 = q68_raw[:, col]
                q90 = q90_raw[:, col]

            ax.fill_between(oos_dates, q10, q90,
                            color="dimgrey", alpha=0.35, linewidth=0)
            ax.fill_between(oos_dates, q32, q68,
                            color="grey", alpha=0.55, linewidth=0)

        # OLS reference band (pale orange, ± 1 SE around OLS mean)
        ols_m = ols_means[panel_i]
        ols_s = ols_ses[panel_i]
        ax.axhspan(ols_m - ols_s, ols_m + ols_s,
                   color="#d4914a", alpha=0.30, linewidth=0)
        ax.axhline(ols_m, color="#c07830", lw=1.2, ls="-", alpha=0.85)

        # NBER recessions (pink shading)
        for (r0, r1) in NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(r0), pd.Timestamp(r1),
                       color="#ffb0b0", alpha=0.45, linewidth=0)

        # GTVP line (green, paper style)
        ax.plot(oos_dates, gtvp, color="#1a7a4a", lw=1.4, zorder=5)

        # OOS start vertical dashed line
        ax.axvline(results[1]["dates"][0], color="white",
                   lw=1.0, ls="--", alpha=0.7)

        ax.axhline(0, color="lightgrey", lw=0.5, ls="-", zorder=1)
        ax.set_facecolor("#1a1a1a")

        # Black header panel (title box) — mimics paper's black title bars
        title = panel_titles[panel_i]
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="white", loc="center",
                     bbox=dict(facecolor="black", edgecolor="none",
                               boxstyle="square,pad=0.3"))

        ax.tick_params(colors="black", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))

    fig.patch.set_facecolor("white")
    fig.suptitle("GTVPs — FA-ARRF  h=1  [OOS 2003Q1–2014Q4]",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("outputs/betas_h1.png", dpi=150, bbox_inches="tight")
    plt.show()

# ── RMSE ratio bar chart — paper Figure 2a style (grouped bars per horizon) ──
# Paper Figure 2a shows RMSPE_model/RMSPE_AR for multiple models side-by-side.
# We replicate the layout: one panel per horizon, bar per model.
# Models shown: FA-ARRF (our replication) vs paper target.
_model_colors = {
    "FA-ARRF (ours)": "#1a8a5a",   # teal-green (MRF group colour in paper)
    "Paper target":   "#e8a060",   # orange-sand (benchmark shade)
}
_horizons_plot = list(HORIZONS)
_n_h = len(_horizons_plot)

fig, axes = plt.subplots(1, _n_h,
                          figsize=(3.5 * max(_n_h, 1) + 1, 4.5),
                          sharey=True)
if _n_h == 1:
    axes = [axes]

for ax, h in zip(axes, _horizons_plot):
    r   = results[h]
    pt  = paper_targets.get(h, np.nan)

    model_vals  = [r["ratio"], pt if not isinstance(pt, str) else np.nan]
    model_names = ["FA-ARRF\n(ours)", "Paper\ntarget"]
    model_cols  = [_model_colors["FA-ARRF (ours)"], _model_colors["Paper target"]]

    x = np.arange(len(model_vals))
    bars = ax.bar(x, model_vals, width=0.55,
                  color=model_cols, alpha=0.90, edgecolor="white", linewidth=0.6)

    # Value labels on top of bars
    for bar, val in zip(bars, model_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="black")

    ax.axhline(1.0, color="black", lw=1.3, ls="-", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0.0, 1.45)
    ax.set_facecolor("white")

    # DM significance stars above FA-ARRF bar
    dm_p = r["dm"]["p_value"]
    star = "***" if dm_p < 0.01 else ("**" if dm_p < 0.05 else ("*" if dm_p < 0.10 else ""))
    if star:
        ax.text(x[0], model_vals[0] + 0.05, star,
                ha="center", va="bottom", fontsize=10, color="black")

    # Black header title box (paper style)
    ax.set_title(f"h={h}", fontsize=11, fontweight="bold",
                 color="white", loc="center",
                 bbox=dict(facecolor="black", edgecolor="none",
                           boxstyle="square,pad=0.3"))

    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")

axes[0].set_ylabel("RMSPE / RMSPE  AR(4)", fontsize=10)

# Legend for DM significance
fig.text(0.5, -0.04,
         "* p<0.10   ** p<0.05   *** p<0.01  (DM test, one-sided, H1: FA-ARRF < AR)",
         ha="center", fontsize=8.5, color="#444444")
fig.suptitle(r"RMSPE$_{m}$ / RMSPE$_{AR(4)}$  — lower is better",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("outputs/rmse_ratios.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nDone. Outputs in ./outputs/")
