import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys
sys.path.insert(0, ".")
from Tree.MRF      import MRF
from experiments.forecast import (build_leading_factor, build_St, create_X_y_S, expanding_window_oos,ar4_benchmark, compute_rmse)
print("Loading data ...")
data = pd.read_excel("fredqd_final_model_ready.xlsx").reset_index(drop=True)
print(f"  Raw: {data.shape}  |  flags: {dict(data['sample_flag'].value_counts())}")
print("Building F_lead (custom leading indicator) ...")
data["F_lead"] = build_leading_factor(data)
data["F_lead_L1"] = data["F_lead"].shift(1)
data["F_lead_L2"] = data["F_lead"].shift(2)
dates  = pd.to_datetime(data["observation_date"])
fl_arr = data["F_lead"].values
print(f"  F_lead 2006Q4: {fl_arr[dates == '2006-10-01'][0]:.3f}  ")
print(f"  F_lead 2007Q3: {fl_arr[dates == '2007-07-01'][0]:.3f}  ")
print(f"  F_lead 2008Q2: {fl_arr[dates == '2008-04-01'][0]:.3f}  ")
print(f"  F_lead 2005Q1: {fl_arr[dates == '2005-01-01'][0]:.3f}  ")
print("Building S_t ...")
S_df, S_col_names, priority_col_idxs = build_St(data, target="UNRATE", n_y_lags=8)
data = data.loc[S_df.index].reset_index(drop=True)
S_df = S_df.reset_index(drop=True)
assert S_df.columns[0] == "trend_t"
print(f"  Rows: {len(data)}  |  S_t cols: {S_df.shape[1]}")
print(f"  Priority vars: {len(priority_col_idxs)} (5× upweighted in mtry)")
HORIZONS     = [1, 2, 4]
UPDATE_FREQ  = 8
results      = {}
beta_dict    = {}
paper_target = {1: 0.82, 2: 0.83, 4: 0.86, 6: 0.88, 8: 0.90}

for h in HORIZONS:
    print(f"\n{'='*65}")
    print(f"  HORIZON h = {h} quarter(s) ahead")
    print(f"{'='*65}")

    X, y, S, flags, dates_X = create_X_y_S(data, S_df, target="UNRATE", h=h)
    oos_mask = flags == "oos"
    print(f"  X: {X.shape}  K={X.shape[1]}  |  OOS: {oos_mask.sum()}  "
          f"|  {dates_X[oos_mask][0].date()} → {dates_X[oos_mask][-1].date()}")

    # AR(4) benchmark
    ar_p, ar_a = ar4_benchmark(data, flags, target="UNRATE", h=h)
    ar_rmse    = compute_rmse(ar_a, ar_p)

    # MRF
    model = MRF(
        n_trees            = 50,
        min_samples_leaf   = 10,
        mtry_frac          = 1/3,
        ridge_lambda       = 0.1,
        rw_regul           = 0.75,
        HRW                = 0.0,
        subsampling_rate   = 0.75,
        block_size         = 8,
        min_leaf_frac_of_x = 1.0,
        no_rw_trespassing  = True,
        max_depth          = 100,
        trend_push         = 2,
        trend_col_idx      = 0,
        fast_rw            = True,
        priority_col_idxs  = priority_col_idxs,
        priority_weight    = 5.0,
        n_jobs             = -1,
    )

    preds, actuals, betas = expanding_window_oos(
        X, y, S, flags, model,
        update_freq = UPDATE_FREQ,
        S_col_names = S_col_names,
        print_diag  = (h == 1),
    )

    mrf_rmse = compute_rmse(actuals, preds)
    ratio    = mrf_rmse / ar_rmse
    beat     = "✓ BEATS AR" if ratio < 1 else "✗ worse"

    results[h]   = dict(mrf_rmse=mrf_rmse, ar_rmse=ar_rmse, ratio=ratio,
                        preds=preds, actuals=actuals,
                        dates=dates_X[oos_mask])
    if h == 1:
        beta_dict[h] = betas

    print(f"\n  h={h}: AR={ar_rmse:.6f}  MRF={mrf_rmse:.6f}  "
          f"ratio={ratio:.4f}  {beat}")
print(f"\n{'='*65}")
print(f"  RESULTS SUMMARY  —  OOS 2003Q1–2014Q4")
print(f"{'='*65}")
print(f"  {'h':>3}  {'AR RMSE':>10}  {'MRF RMSE':>10}  "
      f"{'MRF/AR':>7}  {'Paper target':>12}  {'Status':>10}")
for h in HORIZONS:
    r  = results[h]
    pt = paper_target.get(h, "N/A")
    ok = "✓" if r["ratio"] < 1 else "✗"
    print(f"  {h:>3}  {r['ar_rmse']:>10.6f}  {r['mrf_rmse']:>10.6f}  "
          f"{r['ratio']:>7.4f}  {str(pt):>12}  {ok}")
os.makedirs("outputs", exist_ok=True)
with open("outputs/results.txt", "w") as f:
    f.write(f"OOS: 2003Q1-2014Q4\n")
    f.write(f"{'h':>3}  {'AR_RMSE':>10}  {'MRF_RMSE':>10}  {'ratio':>7}\n")
    for h in HORIZONS:
        r = results[h]
        f.write(f"{h:>3}  {r['ar_rmse']:>10.6f}  "
                f"{r['mrf_rmse']:>10.6f}  {r['ratio']:>7.4f}\n")

RECESSION = (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"))
fig, axes = plt.subplots(
    1, len(HORIZONS),
    figsize=(4 * len(HORIZONS), 5),
    sharey=True,
)
if len(HORIZONS) == 1:
    axes = [axes]

for ax, h in zip(axes, HORIZONS):
    r = results[h]
    act_plot  = r["actuals"] 
    pred_plot = r["preds"]   
    ax.plot(r["dates"], act_plot,  color="black",      lw=1.5, label="Realized")
    ax.plot(r["dates"], pred_plot, color="darkorange",  lw=1.5, label="FA-ARRF")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"),
               color="red", alpha=0.10, label="Recession")
    ax.set_xlim(pd.Timestamp("2003-01-01"), pd.Timestamp("2014-12-31"))
    ax.set_ylim(-3,3)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"h={h}  (MRF/AR={r['ratio']:.3f})", fontsize=10)
axes[0].set_ylabel("Unemployment Rate %)", fontsize=10)
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle(
    f"FA-ARRF Unemployment Rate forecasts vs Realized  ",
    fontsize=12, y=1.02,
)
fig.tight_layout()
fig.savefig("outputs/paper_style_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: outputs/paper_style_plot.png")
if 1 in beta_dict:
    betas = beta_dict[1]
    od    = results[1]["dates"]
    labels = ["μ_t  —  intercept (regime level)",
               "φ₁_t  —  AR lag 1", "φ₂_t  —  AR lag 2",
               "φ₃_t  —  AR lag 3", "φ₄_t  —  AR lag 4",
               "γ₁_t  —  F1 (real activity)",
               "γ₂_t  —  F_lead (leading composite)"]
    K = betas.shape[1]
    fig, axes = plt.subplots(K, 1, figsize=(13, 2.5*K), sharex=True)
    for i, ax in enumerate(axes):
        lbl = labels[i] if i < len(labels) else f"β{i}"
        ax.plot(od, betas[:, i], lw=1.3, color=f"C{i % 10}")
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvspan(*RECESSION, alpha=0.08, color="red")
        ax.set_ylabel(lbl, fontsize=7)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].set_xlabel("Date")
    fig.suptitle("GTVPs  —  FA-ARRF h=1  [OOS 2003Q1–2014Q4]", y=1.01)
    fig.tight_layout()
    fig.savefig("outputs/betas_h1.png", dpi=150, bbox_inches="tight")
    plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
hs     = list(HORIZONS)
ratios = [results[h]["ratio"] for h in hs]
paper  = [paper_target.get(h, np.nan) for h in hs]
x      = np.arange(len(hs));  w = 0.35
ax.bar(x-w/2, ratios, w, label="Our MRF",     color="darkorange", alpha=0.85)
ax.bar(x+w/2, paper,  w, label="Paper target", color="steelblue",  alpha=0.55)
ax.axhline(1.0, color="black", lw=1.2, ls="--", label="AR(4) baseline")
ax.set_xticks(x);  ax.set_xticklabels([f"h={h}" for h in hs])
ax.set_ylabel("RMSE / AR(4) RMSE");  ax.set_title("MRF vs AR(4)  —  lower = better")
ax.legend();  ax.set_ylim(0.5, 1.4)
fig.tight_layout()
fig.savefig("outputs/rmse_ratios.png", dpi=150)
plt.show()
