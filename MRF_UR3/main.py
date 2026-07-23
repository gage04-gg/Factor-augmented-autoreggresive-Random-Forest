import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
sys.path.insert(0, ".")
from Tree.MRF import MRF
from experiments.forecast import (
    build_St,
    create_X_y_S,
    expanding_window_oos,
    ar4_benchmark,
    compute_rmse,
)

HORIZONS    = [1]          # forecast horizons (quarters ahead)
UPDATE_FREQ = 8               # refit every 8 OOS steps (2 years of quarters)
OOS_LABEL   = "2003Q1–2014Q4"

# Paper Table 4 targets for comparison
PAPER_TARGETS = {1: 0.86, 2: 0.83, 4: 0.86, 6: 0.88, 8: 0.90}

# MRF hyperparameters — paper App A.6, quarterly defaults
MRF_KWARGS = dict(
    n_trees            = 100,     # ≥100 for stable mean (paper Algorithm 1)
    min_samples_leaf   = 10,      # Minimal Node Size — paper App A.6
    mtry_frac          = 1 / 3,   # paper default; robust to {0.1, 0.2, 0.33, 0.5}
    ridge_lambda       = 0.1,     # RL (lambda) — paper App A.6
    rw_regul           = 0.1,    # RWR (zeta) — paper App A.6
    HRW                = 0.0,     # pure leaf ridge (paper default)
    subsampling_rate   = 0.75,    # paper App A.6
    block_size         = 8,       # 2-year blocks for quarterly data (paper App A.4)
    min_leaf_frac_of_x = 1.0,     # MLF=1 OK when RWR+RL active (paper App A.6)
    no_rw_trespassing  = True,    # always on (paper App A.6)
    max_depth          = 100,     # effectively unlimited
    trend_push         = 2.0,     # push trend prob above 1/dim(S) (paper App A.6)
    trend_col_idx      = 0,       # trend_t is column 0 of S_t (enforced in build_St)
    fast_rw            = True,    # skip RW during split search (GitHub default)
    n_jobs             = -1,      # use all CPU cores
    # priority_col_idxs and priority_weight set after build_St (data-dependent)
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  FA-ARRF GDP Forecasting Replication")
print("=" * 65)
print("\nLoading FRED-QD data ...")

data = pd.read_excel("fredqd_final_model_ready.xlsx").reset_index(drop=True)
print(f"  Loaded: {data.shape[0]} rows × {data.shape[1]} cols")
print(f"  Sample flags: {dict(data['sample_flag'].value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build S_t
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding S_t ...")
S_df, S_col_names, priority_col_idxs = build_St(
    data, target="UNRATE", n_y_lags=8
)
data = data.loc[S_df.index].reset_index(drop=True)
S_df = S_df.reset_index(drop=True)

assert S_df.columns[0] == "trend_t", (
    f"Expected trend_t as column 0 of S_t, got {S_df.columns[0]}. "
    "Check build_St() column ordering."
)

print(f"  S_t shape: {S_df.shape}  ({S_df.shape[1]} features)")
print(f"  Priority vars found: {len(priority_col_idxs)}")
print(f"  Data rows after alignment: {len(data)}")

# Add priority info to MRF kwargs (data-dependent)
MRF_KWARGS["priority_col_idxs"] = priority_col_idxs
MRF_KWARGS["priority_weight"]   = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. Forecasting loop over horizons
# ─────────────────────────────────────────────────────────────────────────────

results   = {}   # h -> dict(mrf_rmse, ar_rmse, ratio, preds, actuals, dates)
beta_dict = {}   # h -> (n_oos, K) GTVPs

for h in HORIZONS:
    print(f"\n{'=' * 65}")
    print(f"  HORIZON h = {h} quarter(s) ahead")
    print(f"{'=' * 65}")

    # ── 3a. Align arrays ──────────────────────────────────────────────────────
    X, y, S, flags, dates = create_X_y_S(
        data, S_df, target="UNRATE", h=h
    )
    oos_mask = flags == "oos"
    n_oos    = oos_mask.sum()

    print(f"  X: {X.shape}  |  S: {S.shape}  |  OOS: {n_oos}")
    if n_oos > 0:
        print(f"  OOS period: "
              f"{pd.Timestamp(dates[oos_mask][0]).date()} → "
              f"{pd.Timestamp(dates[oos_mask][-1]).date()}")

    # ── 3b. AR(4) benchmark ───────────────────────────────────────────────────
    print(f"\n  Computing AR(4) benchmark ...")
    ar_preds, ar_actuals = ar4_benchmark(
        data, flags, target="UNRATE", h=h
    )
    ar_rmse = compute_rmse(ar_actuals, ar_preds)
    print(f"  AR(4) RMSE = {ar_rmse:.6f}")

    # ── 3c. FA-ARRF (MRF) ─────────────────────────────────────────────────────
    print(f"\n  Running FA-ARRF expanding-window forecast ...")
    model = MRF(**MRF_KWARGS)

    preds, actuals, betas = expanding_window_oos(
        X, y, S, flags, model,
        update_freq  = UPDATE_FREQ,
        S_col_names  = S_col_names,
        print_diag   = (h == 1),      # diagnostics only for first horizon
    )

    mrf_rmse = compute_rmse(actuals, preds)
    ratio    = mrf_rmse / ar_rmse
    pt       = PAPER_TARGETS.get(h, None)
    beat_ar  = "BEATS AR ✓" if ratio < 1.0 else "WORSE THAN AR ✗"
    beat_pt  = (f"| target≈{pt:.2f} {'✓' if ratio <= pt else '✗'}"
                if pt is not None else "")

    print(f"\n  h={h}: AR RMSE={ar_rmse:.6f}  "
          f"MRF RMSE={mrf_rmse:.6f}  "
          f"MRF/AR={ratio:.4f}  {beat_ar} {beat_pt}")

    results[h] = dict(
        mrf_rmse = mrf_rmse,
        ar_rmse  = ar_rmse,
        ratio    = ratio,
        preds    = preds,
        actuals  = actuals,
        dates    = dates[oos_mask],
    )
    if h == 1:
        beta_dict[h] = betas

# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary table
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print(f"  RESULTS SUMMARY  —  OOS {OOS_LABEL}")
print(f"{'=' * 65}")
print(f"  {'h':>4}  {'AR RMSE':>10}  {'MRF RMSE':>10}  "
      f"{'MRF/AR':>8}  {'Paper target':>14}  Status")
for h in HORIZONS:
    r  = results[h]
    pt = PAPER_TARGETS.get(h, "N/A")
    pt_str = f"≈{pt:.2f}" if isinstance(pt, float) else str(pt)
    ok = "✓" if r["ratio"] < 1.0 else "✗"
    print(f"  {h:>4}  {r['ar_rmse']:>10.6f}  {r['mrf_rmse']:>10.6f}  "
          f"{r['ratio']:>8.4f}  {pt_str:>14}  {ok}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save outputs
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

# ── 5a. Text results ──────────────────────────────────────────────────────────
with open("outputs/results.txt", "w") as f:
    f.write(f"FA-ARRF GDP Forecasting — OOS: {OOS_LABEL}\n")
    f.write(f"{'h':>4}  {'AR_RMSE':>10}  {'MRF_RMSE':>10}  {'ratio':>8}\n")
    for h in HORIZONS:
        r = results[h]
        f.write(f"{h:>4}  {r['ar_rmse']:>10.6f}  "
                f"{r['mrf_rmse']:>10.6f}  {r['ratio']:>8.4f}\n")

# ── 5b. Multi-horizon forecast plot (paper style) ────────────────────────────
fig, axes = plt.subplots(
    1, len(HORIZONS),
    figsize=(4 * len(HORIZONS), 5),
    sharey=True,
)
if len(HORIZONS) == 1:
    axes = [axes]

for ax, h in zip(axes, HORIZONS):
    r = results[h]
    # Scale to annualised % (GDP growth stored as fraction → ×400)
    act_plot  = r["actuals"] 
    pred_plot = r["preds"]   

    ax.plot(r["dates"], act_plot,  color="black",      lw=1.5, label="Realized")
    ax.plot(r["dates"], pred_plot, color="darkorange",  lw=1.5, label="FA-ARRF")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    # 2007-09 recession shading
    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"),
               color="red", alpha=0.10, label="Recession")
    ax.set_xlim(pd.Timestamp("2003-01-01"), pd.Timestamp("2014-12-31"))
    ax.set_ylim(-10, 8)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"h={h}  (MRF/AR={r['ratio']:.3f})", fontsize=10)

axes[0].set_ylabel("GDP growth (annualised %)", fontsize=10)
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle(
    f"FA-ARRF GDP forecasts vs Realized  [{OOS_LABEL}]",
    fontsize=12, y=1.02,
)
fig.tight_layout()
fig.savefig("outputs/paper_style_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: outputs/paper_style_plot.png")

# ── 5c. GTVP plot for h=1 ────────────────────────────────────────────────────
if 1 in beta_dict and beta_dict[1] is not None:
    betas     = beta_dict[1]
    oos_dates = results[1]["dates"]
    # K=5: [intercept, phi1, phi2, gamma1(F1), gamma2(F2)]
    labels = [
        r"$\mu_t$  (intercept)",
        r"$\phi_{1,t}$  (AR lag 1)",
        r"$\phi_{2,t}$  (AR lag 2)",
        r"$\gamma_{1,t}$  (F1 real activity)",
        r"$\gamma_{2,t}$  (F2 forward-looking)",
    ]
    n_betas = min(betas.shape[1], len(labels))
    fig, axes = plt.subplots(n_betas, 1, figsize=(13, 2.5 * n_betas), sharex=True)
    if n_betas == 1:
        axes = [axes]
    for i, ax in enumerate(axes[:n_betas]):
        ax.plot(oos_dates, betas[:, i], lw=1.3, color=f"C{i}", label=labels[i])
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"),
                   alpha=0.10, color="red")
        ax.set_ylabel(labels[i], fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"GTVPs — FA-ARRF h=1  [{OOS_LABEL}]", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig("outputs/betas_h1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/betas_h1.png")

# ── 5d. RMSE ratio bar chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, 3 * len(HORIZONS)), 4))
hs      = list(HORIZONS)
ratios  = [results[h]["ratio"] for h in hs]
paper_r = [PAPER_TARGETS.get(h, np.nan) for h in hs]
x       = np.arange(len(hs))
w       = 0.35
ax.bar(x - w / 2, ratios,  w, label="Our MRF",      color="darkorange", alpha=0.85)
ax.bar(x + w / 2, paper_r, w, label="Paper target",  color="steelblue",  alpha=0.55)
ax.axhline(1.0, color="black", lw=1.2, ls="--", label="AR(4) = 1.0")
ax.set_xticks(x)
ax.set_xticklabels([f"h={h}" for h in hs])
ax.set_ylabel("RMSE / AR(4) RMSE")
ax.set_title("MRF vs AR(4) — lower is better")
ax.legend()
ax.set_ylim(0.4, 1.4)
fig.tight_layout()
fig.savefig("outputs/rmse_ratios.png", dpi=150)
plt.close(fig)
print("  Saved: outputs/rmse_ratios.png")

print(f"\n{'=' * 65}")
print("  Done.")
print(f"{'=' * 65}")
