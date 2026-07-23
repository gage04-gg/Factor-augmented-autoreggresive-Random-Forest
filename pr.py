"""
FRED-QD MRF Data Preparation Pipeline
======================================
Goulet Coulombe (2024) — "The Macroeconomy as a Random Forest"

This script takes the raw FRED-QD CSV and produces a single Excel workbook
with two sheets:

  Sheet 1 — PREDICTORS  (245 stationary series, transformed per McCracken & Ng 2020)
            CPI (CPIAUCSL) is overridden to use code 5 (Δln) instead of code 6 (Δ²ln)
            because the paper treats CPI as I(1) for both targets and predictors.

  Sheet 2 — TARGETS     (25 columns = 5 variables × 5 horizons)
            Each column TARGET_hH contains the h-step-ahead direct forecast target
            at row index t, meaning the value observed H periods AFTER t.
            This allows you to directly regress: y_TARGET_hH[t] ~ X[t-H]

Target variables and their transformations (Section 3, paper):
    GDP    → Δln(GDPC1)              I(1), log first-difference
    UR     → ΔUNRATE                 I(1), first-difference
    CPI    → Δln(CPIAUCSL)           I(1), log first-difference  ← NOT code 6
    IR     → ΔGS1                    I(1), first-difference
    SPREAD → GS10TB3Mx (levels)      kept in levels, no transformation

Forecast horizons h = 1, 2, 4, 6, 8 quarters ahead.

Target column naming: GDP_h1, GDP_h2, ..., SPREAD_h8
Row index t of TARGET_hH = the realized value at quarter t+H
(i.e., at forecast origin t, the future outcome you are predicting is in row t)

Usage:
    python fredqd_mrf_prep.py --input 2025-12-QD.csv --output fredqd_MRF_ready.xlsx
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd


# ── Transformation codes (McCracken & Ng 2020) ───────────────────────────────
def transform_series(x: pd.Series, tcode: int, col: str = "") -> pd.Series:
    """Apply FRED-QD transformation code to a raw level series."""
    if   tcode == 1: return x.copy()
    elif tcode == 2: return x.diff(1)
    elif tcode == 3: return x.diff(1).diff(1)
    elif tcode == 4:
        s = x.copy(); s[s <= 0] = np.nan; return np.log(s)
    elif tcode == 5:
        s = x.copy(); s[s <= 0] = np.nan; return np.log(s).diff(1)
    elif tcode == 6:
        s = x.copy(); s[s <= 0] = np.nan; return np.log(s).diff(1).diff(1)
    elif tcode == 7: return (x / x.shift(1)) - 1.0
    else:
        warnings.warn(f"[{col}] Unknown tcode {tcode}, returning raw.", RuntimeWarning)
        return x.copy()


# ── Load raw FRED-QD CSV ──────────────────────────────────────────────────────
def load_raw(filepath: str):
    """
    Returns:
        raw_data    : DataFrame of raw levels (DatetimeIndex, all numeric)
        tcodes      : Series of transform codes indexed by column name
        fcodes      : Series of S&W factor flags (0/1) indexed by column name
    """
    df = pd.read_csv(filepath, header=0, low_memory=False)

    # Extract metadata rows by label
    fcodes = df[df["sasdate"].astype(str).str.strip().str.lower() == "factors"].iloc[0].drop("sasdate")
    tcodes = df[df["sasdate"].astype(str).str.strip().str.lower() == "transform"].iloc[0].drop("sasdate")
    fcodes = pd.to_numeric(fcodes, errors="coerce")
    tcodes = pd.to_numeric(tcodes, errors="coerce")

    # Data rows
    data = df[~df["sasdate"].isin(["factors", "transform"])].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"], format="%m/%d/%Y", errors="coerce")
    data = data.dropna(subset=["sasdate"]).set_index("sasdate").sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")

    return data, tcodes, fcodes


# ── Sheet 1: Predictor panel ──────────────────────────────────────────────────
def build_predictor_panel(raw_data: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    """
    Transform all 245 series using FRED-QD benchmark codes.

    OVERRIDE: CPIAUCSL is forced to code 5 (Δln) instead of code 6 (Δ²ln).
    Reason: The paper (Section 3) treats CPI as I(1) — log first-difference —
    for both the target variable AND as a predictor. Using code 6 would
    over-difference it and remove the inflation signal from the predictor panel.
    """
    results = {}
    overrides = {}

    for col in raw_data.columns:
        if col not in tcodes.index or pd.isna(tcodes[col]):
            warnings.warn(f"[{col}] No transform code — skipping.", RuntimeWarning)
            continue

        tcode = int(tcodes[col])

        # ── OVERRIDE: CPI predictor uses code 5 not code 6 ────────────────
        if col == "CPIAUCSL" and tcode == 6:
            tcode = 5
            overrides[col] = (6, 5)

        results[col] = transform_series(raw_data[col], tcode, col)

    transformed = pd.DataFrame(results, index=raw_data.index)

    # Drop the first 2 rows: code 6 loses 2 obs; code 5 loses 1.
    # Dropping 2 ensures all series are properly initialised.
    transformed = transformed.iloc[2:]

    if overrides:
        print(f"\n  Predictor override applied:")
        for col, (old, new) in overrides.items():
            print(f"    {col}: code {old} → code {new}  (Δ²ln → Δln, per paper Section 3)")

    print(f"\n  Predictor panel: {transformed.shape[0]} quarters × {transformed.shape[1]} series")
    print(f"  Date range: {transformed.index[0].strftime('%YQ%q')} → {transformed.index[-1].strftime('%YQ%q')}")

    return transformed


# ── Sheet 2: Target variables at each horizon ─────────────────────────────────
def build_targets(raw_data: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """
    Construct 5 × len(horizons) direct forecast target columns.

    For I(1) variables, the h-step direct target is the h-period change:
        y^(h)_t  = Y_t − Y_{t−h}   (in log-space for GDP and CPI)

    The column TARGET_hH at row index t contains the value OBSERVED AT t.
    To use in a regression at forecast origin τ = t − H, align with X[τ].

    Column naming: GDP_h1, GDP_h2, GDP_h4, GDP_h6, GDP_h8
                   UR_h1  ...  SPREAD_h8

    SPREAD is kept in levels at all horizons (no differencing), consistent
    with the paper's statement that SPREAD is I(0) in levels.
    """

    # Pre-compute base series from raw levels
    gdp_log    = np.log(raw_data["GDPC1"])         # log GDP
    ur_raw     = raw_data["UNRATE"]                 # unemployment rate (level)
    cpi_log    = np.log(raw_data["CPIAUCSL"])       # log CPI
    ir_raw     = raw_data["GS1"]                    # 1-yr treasury (level)
    spread_raw = raw_data["GS10TB3Mx"]              # 10yr−FFR spread (level)

    targets = {}

    for h in horizons:
        # GDP: Δ_h ln(GDP) = ln(GDP_t) - ln(GDP_{t-h})
        targets[f"GDP_h{h}"]    = gdp_log.diff(h)

        # UR: Δ_h UNRATE = UNRATE_t - UNRATE_{t-h}
        targets[f"UR_h{h}"]     = ur_raw.diff(h)

        # CPI: Δ_h ln(CPI) = ln(CPI_t) - ln(CPI_{t-h})
        # Paper Section 3: "CPI considered I(1), log applied before differencing"
        targets[f"CPI_h{h}"]    = cpi_log.diff(h)

        # IR: Δ_h GS1 = GS1_t - GS1_{t-h}
        targets[f"IR_h{h}"]     = ir_raw.diff(h)

        # SPREAD: kept in levels at all horizons — no differencing
        targets[f"SPREAD_h{h}"] = spread_raw.copy()

    target_df = pd.DataFrame(targets, index=raw_data.index)

    # Drop first 8 rows (max horizon requires 8 lags to initialise)
    target_df = target_df.iloc[8:]

    print(f"\n  Target panel: {target_df.shape[0]} quarters × {target_df.shape[1]} columns")
    print(f"  Date range: {target_df.index[0].strftime('%YQ%q')} → {target_df.index[-1].strftime('%YQ%q')}")
    print(f"  Columns: {list(target_df.columns)}")

    return target_df


# ── Validation ────────────────────────────────────────────────────────────────
def validate(predictors: pd.DataFrame, targets: pd.DataFrame, raw_data: pd.DataFrame):
    """Print a concise sanity-check table."""
    print("\n── Validation ──────────────────────────────────────────────────────")

    # GDP h=1 should approximate ~0.7% quarterly growth
    gdp_h1 = targets["GDP_h1"].dropna()
    print(f"  GDP_h1  mean={gdp_h1.mean()*100:+.4f}%  std={gdp_h1.std()*100:.4f}%"
          f"  (expect ~+0.74%  std ~1.06%)")

    # UR h=1 should be small changes around 0
    ur_h1 = targets["UR_h1"].dropna()
    print(f"  UR_h1   mean={ur_h1.mean():+.4f} pp  std={ur_h1.std():.4f} pp"
          f"  (expect ~0  std ~0.71)")

    # CPI h=1 quarterly inflation ~0.9%
    cpi_h1 = targets["CPI_h1"].dropna()
    print(f"  CPI_h1  mean={cpi_h1.mean()*100:+.4f}%  std={cpi_h1.std()*100:.4f}%"
          f"  (expect ~+0.91%  std ~0.75%)")

    # CPI predictor should be code 5 (smaller std than code 6)
    cpi_pred = predictors["CPIAUCSL"].dropna()
    print(f"  CPI predictor (code 5)  mean={cpi_pred.mean()*100:+.6f}%  "
          f"std={cpi_pred.std()*100:.6f}%")

    # SPREAD in levels
    spread_h1 = targets["SPREAD_h1"].dropna()
    print(f"  SPREAD_h1 (levels)  mean={spread_h1.mean():+.4f}%  "
          f"range=[{spread_h1.min():.2f}, {spread_h1.max():.2f}]"
          f"  (expect ~1.39, range ~-1.5 to 3.8)")

    # GDP h=4 annualised ~2.9%
    gdp_h4 = targets["GDP_h4"].dropna()
    print(f"  GDP_h4  mean={gdp_h4.mean()*100:+.4f}%  (expect ~+2.96% = 4×0.74%)")

    # Check no infinite values
    pred_inf = np.isinf(predictors.values).sum()
    targ_inf = np.isinf(targets.values).sum()
    print(f"\n  Infinite values — predictors: {pred_inf}  targets: {targ_inf}")

    # OOS window check
    oos_preds = predictors.loc["2003-01-01":"2014-12-31"]
    oos_targs = targets.loc["2003-01-01":"2014-12-31"]
    print(f"  OOS window (2003Q1–2014Q4) predictor rows: {len(oos_preds)}")
    print(f"  OOS window (2003Q1–2014Q4) target rows:    {len(oos_targs)}")


# ── Save to Excel ─────────────────────────────────────────────────────────────
def save_excel(predictors: pd.DataFrame, targets: pd.DataFrame, out_path: str):
    """Write two-sheet Excel workbook."""
    print(f"\n  Writing Excel workbook → {out_path}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        predictors.to_excel(writer, sheet_name="Predictors", index=True)
        targets.to_excel(writer, sheet_name="Targets", index=True)

    print("  Done.")


# ── Also save as CSV for easy loading in Python/R ────────────────────────────
def save_csvs(predictors: pd.DataFrame, targets: pd.DataFrame,
              pred_path: str, targ_path: str):
    predictors.to_csv(pred_path)
    targets.to_csv(targ_path)
    print(f"  Predictor CSV → {pred_path}")
    print(f"  Target CSV    → {targ_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build MRF-ready predictor and target datasets from FRED-QD."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Raw FRED-QD CSV (e.g. 2025-12-QD.csv)")
    parser.add_argument("--output", "-o", default="fredqd_MRF_ready.xlsx",
                        help="Output Excel workbook (default: fredqd_MRF_ready.xlsx)")
    parser.add_argument("--pred-csv",  default="fredqd_predictors.csv")
    parser.add_argument("--targ-csv",  default="fredqd_targets.csv")
    args = parser.parse_args()

    horizons = [1, 2, 4, 6, 8]   # paper Section 3

    print("=" * 65)
    print("  FRED-QD MRF Data Preparation Pipeline")
    print("  Goulet Coulombe (2024) — The Macroeconomy as a Random Forest")
    print("=" * 65)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {args.input} ...")
    try:
        raw_data, tcodes, fcodes = load_raw(args.input)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"      Raw data: {raw_data.shape[0]} quarters × {raw_data.shape[1]} series")

    # ── 2. Build predictor panel ──────────────────────────────────────────────
    print("\n[2/4] Building predictor panel ...")
    predictors = build_predictor_panel(raw_data, tcodes)

    # ── 3. Build target variables ─────────────────────────────────────────────
    print("\n[3/4] Building target variables ...")
    print(f"      Horizons h = {horizons}")
    print(f"      Targets: GDP (Δln), UR (Δ), CPI (Δln), IR (Δ), SPREAD (levels)")
    targets = build_targets(raw_data, horizons)

    # ── 4. Validate & Save ────────────────────────────────────────────────────
    print("\n[4/4] Validating and saving ...")
    validate(predictors, targets, raw_data)
    save_excel(predictors, targets, args.output)
    save_csvs(predictors, targets, args.pred_csv, args.targ_csv)

    print("\n" + "=" * 65)
    print("  Output files:")
    print(f"    {args.output}  — Excel workbook (2 sheets)")
    print(f"    {args.pred_csv}  — predictor panel CSV")
    print(f"    {args.targ_csv}    — target variables CSV")
    print()
    print("  Sheet 1 — Predictors (245 cols):")
    print("    • All FRED-QD series transformed per McCracken & Ng (2020)")
    print("    • CPIAUCSL overridden: code 6 → code 5 (Δln, per paper)")
    print()
    print("  Sheet 2 — Targets (25 cols = 5 vars × 5 horizons):")
    print("    • GDP_h{H}    = ln(GDP_t) − ln(GDP_{t-H})")
    print("    • UR_h{H}     = UNRATE_t − UNRATE_{t-H}")
    print("    • CPI_h{H}    = ln(CPI_t) − ln(CPI_{t-H})")
    print("    • IR_h{H}     = GS1_t − GS1_{t-H}")
    print("    • SPREAD_h{H} = GS10TB3Mx_t  (levels, no differencing)")
    print()
    print("  How to use targets in regression:")
    print("    At forecast origin t, predict TARGET_hH[t+H] using X[t]")
    print("    i.e., target.shift(-H) aligns the future value with current X")
    print("=" * 65)

    return predictors, targets


if __name__ == "__main__":
    main()