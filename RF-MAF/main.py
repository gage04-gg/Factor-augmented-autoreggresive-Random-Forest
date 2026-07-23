import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys

sys.path.insert(0, ".")

# ✅ RF-MAF model
from Tree.rmf import RF_MAF_Forest

# ✅ Updated functions
from experiments.forecast import (
    build_St,
    create_y_S,
    expanding_window_oos_ar_rf,
    ar4_benchmark,
    compute_rmse
)

print("Loading data ...")
data = pd.read_excel("fredqd_final_model_ready.xlsx").reset_index(drop=True)

print(f"Raw: {data.shape} | flags: {dict(data['sample_flag'].value_counts())}")

# ─────────────────────────────────────────────
# Build S_t
# ─────────────────────────────────────────────
print("Building S_t ...")
S_df, S_col_names, priority_col_idxs = build_St(data, target="GDPC1", n_y_lags=8)

data = data.loc[S_df.index].reset_index(drop=True)
S_df = S_df.reset_index(drop=True)

print(f"Rows: {len(data)} | S_t cols: {S_df.shape[1]}")
print(f"Priority vars: {len(priority_col_idxs)}")

# ─────────────────────────────────────────────
HORIZONS = [1, 2]
UPDATE_FREQ = 8

results = {}

# ─────────────────────────────────────────────
for h in HORIZONS:

    print("\n" + "="*60)
    print(f"HORIZON h = {h}")
    print("="*60)

    # ✅ FIXED
    y, S, flags, dates = create_y_S(data, S_df, target="GDPC1", h=h)

    oos_mask = flags == "oos"

    print(f"S: {S.shape} | OOS: {oos_mask.sum()}")

    # ── AR(4) benchmark ───────────────────────
    ar_preds, ar_actuals = ar4_benchmark(data, flags, target="GDPC1", h=h)
    ar_rmse = compute_rmse(ar_actuals, ar_preds)

    # ── RF-MAF model ──────────────────────────
    model = RF_MAF_Forest(
        n_trees=50,
        min_samples_leaf=10,
        mtry_frac=0.2,
        subsampling_rate=0.75,
        block_size=12,
        max_depth=10,
    )

    preds, actuals = expanding_window_oos_ar_rf(
        y, S, flags, model,
        update_freq=UPDATE_FREQ
    )

    rmse = compute_rmse(actuals, preds)
    ratio = rmse / ar_rmse

    results[h] = dict(
        rmse=rmse,
        ar_rmse=ar_rmse,
        ratio=ratio,
        preds=preds,
        actuals=actuals,
        dates=dates[oos_mask]
    )

    print(f"h={h}: AR={ar_rmse:.6f} | RF-MAF={rmse:.6f} | ratio={ratio:.4f}")

# ─────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

for h in HORIZONS:
    r = results[h]
    print(f"h={h} | AR={r['ar_rmse']:.6f} | RF-MAF={r['rmse']:.6f} | ratio={r['ratio']:.4f}")

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4*len(HORIZONS), 4), sharey=True)

if len(HORIZONS) == 1:
    axes = [axes]

for ax, h in zip(axes, HORIZONS):
    r = results[h]

    actuals_plot = r["actuals"] * 400
    preds_plot   = r["preds"] * 400

    ax.plot(r["dates"], actuals_plot, color="black", lw=1.5, label="Actual")
    ax.plot(r["dates"], preds_plot, color="darkorange", lw=1.5, label="RF-MAF")

    ax.axhline(0, color="grey", lw=0.6, ls="--")

    ax.axvspan(pd.Timestamp("2007-12-01"),
               pd.Timestamp("2009-06-01"),
               color="red", alpha=0.1)

    ax.set_xlim(pd.Timestamp("2003-01-01"),
                pd.Timestamp("2014-12-31"))

    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))

    ax.set_title(f"h={h}")

axes[0].set_ylabel("GDP growth (annualized %)")
axes[0].legend()

plt.tight_layout()
plt.savefig("outputs/rf_maf_plot.png", dpi=150)
plt.show()

