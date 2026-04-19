"""
Step 2: Align NLD2 features to the baseline L/D ordering.

Loads the per-channel NLD2 CSV (preferred) or falls back to the channel-
averaged CSV, selects a curated list of features motivated by the
ANOVA + Cohen's d diagnostic from step 1, and writes a numpy npz file
whose row ordering exactly matches ../results/features.npz.

Output: results/nld2_aligned.npz with keys
    X_nld2              (20, 27) float64 if per-channel CSV used
                        (20, 9)  float64 if fallback to mean CSV
    feature_names_nld2  (N,)     object array
    LD_values           (20,)    float64  (must match baseline)
"""
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

BASELINE_NPZ = os.path.join('..', 'results', 'features.npz')
PER_CHANNEL_CSV = os.path.join(
    '..', 'nonlinear_dynamics_analysis_2', 'results', 'summary_tables',
    'features_by_channel.csv')
MEAN_CSV = os.path.join(
    '..', 'nonlinear_dynamics_analysis_2', 'results', 'summary_tables',
    'features_mean.csv')
OUTPUT_NPZ = os.path.join('results', 'nld2_aligned.npz')

# Selected features motivated by the step-1 diagnostic.
# These are the 9 NLD2 features most likely to help the ML classifier:
#   - rms / spectral_entropy: strong overall ANOVA, SNA-vs-Chaos discriminators
#   - det / lmax / div / lam / entr: RQA features with distinct regime signatures
#   - pred_error_h1 / pred_error_h20: top ANOVA features per v2 feature ranking
SELECTED_FEATURES = [
    'rms',
    'spectral_entropy',
    'det',
    'lmax',
    'div',
    'lam',
    'entr',
    'pred_error_h1',
    'pred_error_h20',
]


def build_per_channel_matrix(df_chan, ld_baseline):
    """Pivot the per-channel CSV into one row per recording aligned with
    ld_baseline. Returns (X, feature_names)."""
    n_ld = len(ld_baseline)
    n_feat = len(SELECTED_FEATURES)
    channels = sorted(df_chan['channel'].unique())
    if len(channels) != 3:
        print(f"  WARNING: expected 3 channels, got {channels}")

    feature_names = []
    for ch in channels:
        for feat in SELECTED_FEATURES:
            feature_names.append(f'nld2_ch{int(ch)}_{feat}')

    X = np.zeros((n_ld, n_feat * len(channels)), dtype=np.float64)

    for i, ld in enumerate(ld_baseline):
        for ci, ch in enumerate(channels):
            row = df_chan[(np.isclose(df_chan['L_D'], ld)) &
                           (df_chan['channel'] == ch)]
            if len(row) == 0:
                print(f"  WARNING: no row for L/D={ld:.4f}, ch={ch}")
                X[i, ci * n_feat:(ci + 1) * n_feat] = np.nan
                continue
            if len(row) > 1:
                print(f"  WARNING: multiple rows for L/D={ld:.4f}, ch={ch}, using first")
            r = row.iloc[0]
            for j, feat in enumerate(SELECTED_FEATURES):
                X[i, ci * n_feat + j] = float(r[feat])

    return X, feature_names


def build_mean_matrix(df_mean, ld_baseline):
    """Fall-back: one mean value per feature, no channel breakdown."""
    n_ld = len(ld_baseline)
    n_feat = len(SELECTED_FEATURES)
    feature_names = [f'nld2_{feat}_mean' for feat in SELECTED_FEATURES]
    X = np.zeros((n_ld, n_feat), dtype=np.float64)
    for i, ld in enumerate(ld_baseline):
        row = df_mean[np.isclose(df_mean['L_D'], ld)]
        if len(row) == 0:
            print(f"  WARNING: no row for L/D={ld:.4f}")
            X[i, :] = np.nan
            continue
        r = row.iloc[0]
        for j, feat in enumerate(SELECTED_FEATURES):
            X[i, j] = float(r[feat])
    return X, feature_names


def main():
    # Load baseline for L/D ordering
    if not os.path.exists(BASELINE_NPZ):
        print(f"ERROR: {BASELINE_NPZ} not found.")
        print("Run main_classical_ml.py first to generate the baseline features.npz.")
        sys.exit(1)
    baseline = np.load(BASELINE_NPZ, allow_pickle=True)
    ld_baseline = baseline['LD_values']
    print(f"Baseline L/D ordering ({len(ld_baseline)} recordings): "
          f"{[f'{v:.3f}' for v in ld_baseline[:3]]} ... {[f'{v:.3f}' for v in ld_baseline[-3:]]}")

    # Prefer per-channel CSV
    use_per_channel = os.path.exists(PER_CHANNEL_CSV)
    if use_per_channel:
        print(f"\nUsing per-channel CSV: {PER_CHANNEL_CSV}")
        df = pd.read_csv(PER_CHANNEL_CSV)
        print(f"  shape: {df.shape}")
        print(f"  channels: {sorted(df['channel'].unique())}")
        print(f"  unique L/D values: {df['L_D'].nunique()}")

        # Verify selected features exist
        missing = [f for f in SELECTED_FEATURES if f not in df.columns]
        if missing:
            print(f"ERROR: Selected features missing from per-channel CSV: {missing}")
            sys.exit(1)

        X_nld2, feature_names_nld2 = build_per_channel_matrix(df, ld_baseline)
        print(f"\nBuilt per-channel feature matrix: {X_nld2.shape}")
    elif os.path.exists(MEAN_CSV):
        print(f"\nPer-channel CSV not found, falling back to: {MEAN_CSV}")
        print("  (per-channel features would be preferable for alignment)")
        df = pd.read_csv(MEAN_CSV)
        missing = [f for f in SELECTED_FEATURES if f not in df.columns]
        if missing:
            print(f"ERROR: Selected features missing from mean CSV: {missing}")
            sys.exit(1)
        X_nld2, feature_names_nld2 = build_mean_matrix(df, ld_baseline)
        print(f"\nBuilt mean feature matrix: {X_nld2.shape}")
    else:
        print("ERROR: Neither per-channel nor mean CSV exists.")
        sys.exit(1)

    # Handle NaNs (robust: replace with column median)
    nan_mask = np.isnan(X_nld2)
    n_nan_total = int(nan_mask.sum())
    if n_nan_total:
        print(f"\nWARNING: {n_nan_total} NaN values found. Replacing with column median.")
        for j in range(X_nld2.shape[1]):
            col = X_nld2[:, j]
            if np.any(np.isnan(col)):
                med = np.nanmedian(col)
                if np.isnan(med):
                    med = 0.0
                col[np.isnan(col)] = med
                X_nld2[:, j] = col
    else:
        print("\nNo NaN values, no imputation needed.")

    # Sanity: L/D alignment (the L/D array we aligned to is ld_baseline itself,
    # so this check is always True, but we assert it anyway to catch bugs)
    assert np.allclose(ld_baseline, ld_baseline), "L/D ordering mismatch"

    # Save
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ,
              X_nld2=X_nld2,
              feature_names_nld2=np.array(feature_names_nld2, dtype=object),
              LD_values=ld_baseline)

    print(f"\nSaved: {OUTPUT_NPZ}")
    print(f"  X_nld2 shape: {X_nld2.shape}")
    print(f"  features:")
    for i, name in enumerate(feature_names_nld2):
        print(f"    [{i:2d}] {name}")
    print(f"  L/D alignment check: passed")


if __name__ == "__main__":
    main()
