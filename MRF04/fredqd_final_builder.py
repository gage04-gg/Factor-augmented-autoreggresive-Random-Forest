"""
═══════════════════════════════════════════════════════════════════════
  FRED-QD  →  Final MRF-Ready Dataset
  Goulet Coulombe (2024) — "The Macroeconomy as a Random Forest"
═══════════════════════════════════════════════════════════════════════

INPUT : fredqd_St_final.xlsx  (St_full + Targets sheets)
OUTPUT: fredqd_final_model_ready.csv   ← single flat file, all cols
        fredqd_final_model_ready.xlsx  ← same, Excel with 3 sheets

WHAT THIS SCRIPT DOES
─────────────────────
1. Load St_full (1271 cols) + Targets (25 cols) — already built
2. Fix date index  → proper 'observation_date' column (YYYY-MM-DD)
   AND a 'quarter' column (YYYYQq) for human readability
3. Handle missing values (two separate strategies):
   a. Leading NaNs  (series not yet available)  → column-mean fill
   b. Trailing NaN at 2025Q3 (38 cols)          → forward-fill (last value)
4. Add y_t lags for each of 5 target variables (8 lags each = 40 cols)
   These complete the S_t per Table 1: "Eight lags of y_t"
5. Add 'sample_flag' column: 'train' / 'oos' / 'post_oos'
6. Merge St with Targets on common index
7. Write final clean CSVs and Excel sheets

FINAL DATASET COLUMNS
─────────────────────
  observation_date   YYYY-MM-DD  (first day of quarter)
  quarter            YYYYQq string
  sample_flag        train | oos | post_oos
  trend_t            integer time trend (1, 2, 3, …)
  [245 raw series]   transformed FRED-QD predictors
  [490 raw lags]     L1 and L2 of each raw series
  [5 factors]        F1–F5 PCA cross-sectional factors
  [40 factor lags]   F1_L1 … F5_L8
  [490 MAFs]         series_MAF1 and series_MAF2 for each series
  [40 y_t lags]      GDP_yL1…GDP_yL8, UR_yL1…SPREAD_yL8
  [25 targets]       GDP_h1…SPREAD_h8

TOTAL: ~1,336 columns ready for MRF / AR(4) / any model
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE   = "fredqd_St_final.xlsx"
OUT_CSV      = "fredqd_final_model_ready.csv"
OUT_XLSX     = "fredqd_final_model_ready.xlsx"

# ── Paper constants ───────────────────────────────────────────────────────────
TRAIN_START  = "1963Q3"    # first quarter after 8-lag MAF burn-in
TRAIN_END    = "2002Q4"
OOS_START    = "2003Q1"
OOS_END      = "2014Q4"

TARGETS      = ["GDP", "UR", "CPI", "IR", "SPREAD"]
HORIZONS     = [1, 2, 4, 6, 8]
N_YT_LAGS    = 8           # "eight lags of y_t" per Table 1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("  FRED-QD → Final MRF-Ready Dataset")
print("=" * 68)

print("\n[1/5] Loading sheets ...")
St = pd.read_excel(INPUT_FILE, sheet_name="St_full",  index_col=0)
tg = pd.read_excel(INPUT_FILE, sheet_name="Targets",  index_col=0)

print(f"  St_full  : {St.shape[0]} rows × {St.shape[1]} cols  "
      f"(index: '{St.index[0]}' … '{St.index[-1]}')")
print(f"  Targets  : {tg.shape[0]} rows × {tg.shape[1]} cols")
print(f"  St NaNs  : {St.isna().sum().sum()} across "
      f"{(St.isna().sum()>0).sum()} columns")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATE INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Fixing date index ...")

def quarter_str_to_date(q: str) -> pd.Timestamp:
    """'2003Q1' → Timestamp('2003-01-01')"""
    year = int(q[:4])
    qnum = int(q[5])
    month = {1: 1, 2: 4, 3: 7, 4: 10}[qnum]
    return pd.Timestamp(year=year, month=month, day=1)

# Build dual index
quarters    = list(St.index)                             # "1963Q3", ...
obs_dates   = [quarter_str_to_date(q) for q in quarters] # Timestamp

# Convert both frames to DatetimeIndex for alignment
St.index = pd.DatetimeIndex(obs_dates)
St.index.name = "observation_date"

tg_quarters = list(tg.index)
tg.index    = pd.DatetimeIndex([quarter_str_to_date(q) for q in tg_quarters])
tg.index.name = "observation_date"

print(f"  St  DatetimeIndex: {St.index[0].date()} → {St.index[-1].date()}")
print(f"  Tgt DatetimeIndex: {tg.index[0].date()} → {tg.index[-1].date()}")

# Verify quarterly spacing
diffs = pd.Series(St.index).diff().dropna().dt.days
assert diffs.between(89, 93).all(), "Non-quarterly spacing detected!"
print(f"  Quarterly spacing verified: {diffs.min()}–{diffs.max()} days per step ✓")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MISSING VALUE HANDLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Handling missing values ...")

nan_cols = (St.isna().sum() > 0)
print(f"  Columns with any NaN: {nan_cols.sum()}")

St_clean = St.copy()

# Strategy A: trailing NaN at last quarter (2025Q3) → forward-fill
#  These arise because L1/L2/MAF lags extend past the last available FRED data
trailing_nans = []
last_q = St_clean.index[-1]
for col in St_clean.columns:
    if pd.isna(St_clean.loc[last_q, col]):
        trailing_nans.append(col)

if trailing_nans:
    St_clean[trailing_nans] = St_clean[trailing_nans].ffill()
    print(f"  Strategy A — forward-fill trailing NaN ({last_q.date()}): "
          f"{len(trailing_nans)} cols fixed")

# Strategy B: leading NaNs (series not available early in sample)
#  → fill with column mean (McCracken & Ng 2002 EM approach)
remaining_nan_cols = [c for c in St_clean.columns if St_clean[c].isna().any()]
for col in remaining_nan_cols:
    col_mean = St_clean[col].mean()          # mean over non-NaN values only
    St_clean[col] = St_clean[col].fillna(col_mean)

print(f"  Strategy B — mean-fill leading NaNs: {len(remaining_nan_cols)} cols fixed")
print(f"  Remaining NaNs: {St_clean.isna().sum().sum()} ✓")
assert St_clean.isna().sum().sum() == 0, "Still have NaNs after imputation!"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ADD y_t LAGS (completing Table 1 Component 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Adding 8 lags of each target variable (Table 1, Component 1) ...")

# For each target variable v at horizon h=1 (the base series),
# we create 8 lags. In the regression at origin t you use these
# as the "eight lags of y_t" that go into S_t.
# We use h=1 series as the base y_t series for each variable.

yt_lags_dict = {}
for v in TARGETS:
    base_col = f"{v}_h1"          # y_t for this variable (1-step version)
    if base_col not in tg.columns:
        print(f"  WARNING: {base_col} not in Targets — skipping y_t lags for {v}")
        continue
    yt = tg[base_col]
    for l in range(1, N_YT_LAGS + 1):
        lag_col = f"{v}_yL{l}"
        yt_lags_dict[lag_col] = yt.shift(l)

yt_lags = pd.DataFrame(yt_lags_dict)
n_ytlag_cols = len(yt_lags_dict)
print(f"  Added {len(TARGETS)} targets × {N_YT_LAGS} lags = {n_ytlag_cols} columns")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MERGE, FLAG, FINALISE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Assembling final dataset ...")

# Common DatetimeIndex
common_idx = (St_clean.index
              .intersection(tg.index)
              .intersection(yt_lags.index))
common_idx = common_idx.sort_values()
print(f"  Common index: {common_idx[0].date()} → {common_idx[-1].date()} "
      f"({len(common_idx)} quarters)")

St_aligned   = St_clean.loc[common_idx]
tg_aligned   = tg.loc[common_idx]
ytl_aligned  = yt_lags.loc[common_idx]

# Build sample flag
train_start_dt = quarter_str_to_date(TRAIN_START)
train_end_dt   = quarter_str_to_date(TRAIN_END)
oos_start_dt   = quarter_str_to_date(OOS_START)
oos_end_dt     = quarter_str_to_date(OOS_END)

def flag(dt):
    if dt <= train_end_dt:  return "train"
    if dt <= oos_end_dt:    return "oos"
    return "post_oos"

sample_flag = pd.Series([flag(d) for d in common_idx],
                         index=common_idx, name="sample_flag")

# Quarter label column
quarter_col = pd.Series(
    [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in common_idx],
    index=common_idx, name="quarter"
)

# Assemble in logical order:
# observation_date | quarter | sample_flag | trend | raw | raw_lags |
# factors | factor_lags | MAFs | y_t_lags | targets
final = pd.concat([
    quarter_col,
    sample_flag,
    St_aligned,
    ytl_aligned,
    tg_aligned,
], axis=1)

final.index.name = "observation_date"

# ── Verification ──────────────────────────────────────────────────────────────
print("\n" + "─" * 68)
print("  VERIFICATION")
print("─" * 68)

train_df = final[final["sample_flag"] == "train"]
oos_df   = final[final["sample_flag"] == "oos"]
post_df  = final[final["sample_flag"] == "post_oos"]

print(f"  Total rows    : {len(final)}")
print(f"  Training rows : {len(train_df)}  "
      f"({train_df.index[0].date()} → {train_df.index[-1].date()})  "
      f"[need ≥ 166]")
print(f"  OOS rows      : {len(oos_df)}   "
      f"({oos_df.index[0].date()} → {oos_df.index[-1].date()})  "
      f"[need = 48]  {'✓' if len(oos_df)==48 else '✗'}")
print(f"  Post-OOS rows : {len(post_df)}")
print(f"  Total columns : {final.shape[1]}")
print(f"  NaN count     : {final.isna().sum().sum()}  "
      f"{'✓' if final.isna().sum().sum()==0 else '← y_t lags have leading NaNs (expected)'}")

# Column breakdown
meta_cols    = ["quarter", "sample_flag"]
trend_cols   = ["trend_t"]
raw_cols     = [c for c in St_aligned.columns
                if "_L" not in c and "_MAF" not in c
                and c not in ["trend_t"] and not c.startswith("F")]
raw_lag_cols = [c for c in St_aligned.columns
                if "_L" in c and not c.startswith("F")]
fac_cols     = [c for c in St_aligned.columns if c in ["F1","F2","F3","F4","F5"]]
fac_lag_cols = [c for c in St_aligned.columns
                if c.startswith("F") and "_L" in c]
maf_cols     = [c for c in St_aligned.columns if "_MAF" in c]
ytlag_cols   = [c for c in ytl_aligned.columns]
targ_cols    = [c for c in tg_aligned.columns]

print()
print("  Column breakdown:")
print(f"    Meta (quarter, sample_flag)  :   {len(meta_cols):>4}")
print(f"    Time trend                   :   {len(trend_cols):>4}")
print(f"    Raw series (245)             :  {len(raw_cols):>4}")
print(f"    Raw lags (×2)                :  {len(raw_lag_cols):>4}")
print(f"    PCA factors F1–F5            :   {len(fac_cols):>4}")
print(f"    Factor lags (×8)             :  {len(fac_lag_cols):>4}")
print(f"    MAFs (245 × 2)               :  {len(maf_cols):>4}")
print(f"    y_t lags (5 vars × 8)        :  {len(ytlag_cols):>4}")
print(f"    Targets (5 vars × 5 horizons):  {len(targ_cols):>4}")
print(f"    {'─'*35}")
print(f"    TOTAL                        : {final.shape[1]:>5}")

# Key value checks
print()
print("  Key value checks:")
print(f"    GDPC1 mean (expect ~0.74%)   : {final['GDPC1'].mean()*100:+.4f}%  "
      f"{'✓' if abs(final['GDPC1'].mean()*100 - 0.74) < 0.5 else '✗'}")
print(f"    UNRATE mean (expect ~0)      : {final['UNRATE'].mean():+.5f}  "
      f"{'✓' if abs(final['UNRATE'].mean()) < 0.1 else '✗'}")
print(f"    GS10TB3Mx mean (expect ~1.39): {final['GS10TB3Mx'].mean():+.4f}  "
      f"{'✓' if abs(final['GS10TB3Mx'].mean() - 1.39) < 0.3 else '✗'}")
print(f"    GDP_h1 mean (expect ~0.74%)  : {final['GDP_h1'].mean()*100:+.4f}%  "
      f"{'✓' if abs(final['GDP_h1'].mean()*100 - 0.74) < 0.5 else '✗'}")
print(f"    OOS rows = 48                : {'✓' if len(oos_df)==48 else '✗'}")
print(f"    Zero NaN in St columns       : "
      f"{'✓' if final[list(St_aligned.columns)].isna().sum().sum()==0 else '✗'}")

# ── Save ──────────────────────────────────────────────────────────────────────
print()
print("─" * 68)
print("  SAVING")
print("─" * 68)

# CSV — flat file, index = observation_date
print(f"  Writing {OUT_CSV} ...")
final.to_csv(OUT_CSV, date_format="%Y-%m-%d")
print(f"  → {final.shape[0]} rows × {final.shape[1]} cols")

# Excel — 3 windows + legend sheet
print(f"  Writing {OUT_XLSX} ...")

train_sheet = final[final["sample_flag"] == "train"].copy()
oos_sheet   = final[final["sample_flag"] == "oos"].copy()
full_sheet  = final.copy()

# Column legend
legend_rows = (
    [("observation_date", "index", "First day of quarter (YYYY-MM-DD)")]
  + [("quarter",          "meta",  "Quarter label e.g. 2003Q1")]
  + [("sample_flag",      "meta",  "train | oos | post_oos")]
  + [("trend_t",          "St",    "Integer time trend t (Table 1, exogenous structural change)")]
  + [(c, "St - raw",      "Transformed FRED-QD predictor (contemporaneous)") for c in raw_cols[:3]]
  + [("...",              "St - raw", f"… {len(raw_cols)-3} more raw series")]
  + [(c, "St - raw_lag",  "Lag of raw predictor (L1 or L2)") for c in raw_lag_cols[:3]]
  + [("...",              "St - raw_lag", f"… {len(raw_lag_cols)-3} more")]
  + [(c, "St - factor",   "Cross-sectional PCA factor") for c in fac_cols]
  + [(c, "St - fac_lag",  "Lag of PCA factor") for c in fac_lag_cols[:3]]
  + [("...",              "St - fac_lag", f"… {len(fac_lag_cols)-3} more")]
  + [(c, "St - MAF",      "Moving-Average Factor (PCA on 8 lags of series j)") for c in maf_cols[:3]]
  + [("...",              "St - MAF", f"… {len(maf_cols)-3} more")]
  + [(c, "St - y_lag",    "Lag of target y_t (Table 1, endogenous SETAR dynamics)") for c in ytlag_cols[:3]]
  + [("...",              "St - y_lag", f"… {len(ytlag_cols)-3} more")]
  + [(c, "Target",        "Direct h-step forecast target") for c in targ_cols]
)
legend_df = pd.DataFrame(legend_rows, columns=["column", "type", "description"])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    full_sheet.to_excel(  writer, sheet_name="Full_1963Q3_2025Q3")
    train_sheet.to_excel( writer, sheet_name="Train_1963Q3_2002Q4")
    oos_sheet.to_excel(   writer, sheet_name="OOS_2003Q1_2014Q4")
    legend_df.to_excel(   writer, sheet_name="Column_Legend", index=False)

print(f"  → 4 sheets written")
print()
print("=" * 68)
print("  DONE — Final model-ready dataset")
print("=" * 68)
print()
print(f"  {OUT_CSV}")
print(f"    → {final.shape[0]} rows × {final.shape[1]} columns")
print(f"    → index = observation_date (YYYY-MM-DD)")
print()
print(f"  {OUT_XLSX}")
print(f"    Sheet 1: Full_1963Q3_2025Q3    — {full_sheet.shape}")
print(f"    Sheet 2: Train_1963Q3_2002Q4   — {train_sheet.shape}")
print(f"    Sheet 3: OOS_2003Q1_2014Q4     — {oos_sheet.shape}")
print(f"    Sheet 4: Column_Legend         — {legend_df.shape}")
print()
print("  HOW TO USE IN REGRESSION (example for GDP, h=4):")
print("  ─────────────────────────────────────────────────")
print("  df = pd.read_csv('fredqd_final_model_ready.csv', index_col=0, parse_dates=True)")
print("  train = df[df.sample_flag == 'train']")
print("  oos   = df[df.sample_flag == 'oos']")
print("  # St columns (features):")
print("  st_cols = [c for c in df.columns if c not in")
print("             ['quarter','sample_flag','GDP_h1','UR_h1',...]]")
print("  # At origin t, predict GDP_h4[t+4] — use .shift(-4) on target:")
print("  y = df['GDP_h4'].shift(-4)")
print("  # Add 8 lags of y_t (already in df as GDP_yL1 … GDP_yL8)")
print("=" * 68)
