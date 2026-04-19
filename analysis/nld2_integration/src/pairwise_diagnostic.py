"""
Pairwise discriminability diagnostic for NLD2 features.

Computes ANOVA F-statistic across all regimes and pairwise |Cohen's d|
for every regime pair, producing:
    - discriminability_table.csv   (features x metrics)
    - discriminability_heatmap.png (top-25 features x 10 regime pairs)
    - sna_vs_chaos_top10.csv       (ranked by |d| on SNA vs Chaos)
    - top_features_per_pair.txt    (top 5 per pair, plain text)
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================================
# Cohen's d (absolute, pooled std with ddof=1)
# =========================================================================

def cohens_d(group_a, group_b):
    """Absolute standardized mean difference. Returns NaN if undefined."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float('nan')
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    denom = n_a + n_b - 2
    if denom <= 0:
        return float('nan')
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / denom)
    if pooled < 1e-12:
        return float('nan')
    return float(abs(np.mean(a) - np.mean(b)) / pooled)


# =========================================================================
# Main diagnostic
# =========================================================================

def analyze_discriminability(features_df, regime_col='regime_name',
                               exclude_cols=None, output_dir='.'):
    """
    For every numeric feature in features_df, compute:
      - Overall ANOVA F and p across regime groups
      - Pairwise |Cohen's d| for every pair of regimes present in the data

    Writes the four output files described in the module docstring into
    output_dir. Returns a dict with the computed DataFrames for reuse.
    """
    if exclude_cols is None:
        exclude_cols = []
    os.makedirs(output_dir, exist_ok=True)

    # Identify numeric feature columns
    feature_cols = [
        c for c in features_df.columns
        if c not in exclude_cols and c != regime_col
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    regimes = sorted(features_df[regime_col].dropna().unique().tolist())
    regime_pairs = list(combinations(regimes, 2))

    rows = []
    for feat in feature_cols:
        row = {'feature': feat}

        # ANOVA across all regimes with >=2 samples
        groups = []
        for r in regimes:
            vals = features_df.loc[features_df[regime_col] == r, feat].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) >= 2:
            try:
                f_stat, p_val = sstats.f_oneway(*groups)
                row['anova_F'] = float(f_stat)
                row['anova_p'] = float(p_val)
            except Exception:
                row['anova_F'] = float('nan')
                row['anova_p'] = float('nan')
        else:
            row['anova_F'] = float('nan')
            row['anova_p'] = float('nan')

        # Pairwise |Cohen's d|
        for r1, r2 in regime_pairs:
            a = features_df.loc[features_df[regime_col] == r1, feat].values
            b = features_df.loc[features_df[regime_col] == r2, feat].values
            col_name = f'd_{r1}_vs_{r2}'
            row[col_name] = cohens_d(a, b)
        rows.append(row)

    disc_df = pd.DataFrame(rows).sort_values('anova_F', ascending=False)
    disc_df.to_csv(os.path.join(output_dir, 'discriminability_table.csv'),
                    index=False)

    # ---- Heatmap: top 25 features x all regime pairs ----
    top_features = disc_df.head(25)['feature'].tolist()
    pair_cols = [f'd_{r1}_vs_{r2}' for r1, r2 in regime_pairs]
    heatmap_data = disc_df[disc_df['feature'].isin(top_features)].set_index('feature')[pair_cols]
    heatmap_data = heatmap_data.reindex(top_features)
    # Clean pair labels for display
    clean_labels = [col.replace('d_', '').replace('_vs_', ' vs ') for col in pair_cols]
    heatmap_data.columns = clean_labels

    fig_w = max(8, 0.9 * len(pair_cols) + 4)
    fig_h = max(7, 0.32 * len(top_features) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = min(8.0, float(np.nanmax(heatmap_data.values)) if heatmap_data.size else 1.0)
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=vmax, cbar_kws={'label': "|Cohen's d|"}, ax=ax,
                linewidths=0.3, linecolor='white')
    ax.set_title("Top 25 NLD2 features: |Cohen's d| across regime pairs",
                 fontsize=12)
    ax.set_xlabel('Regime pair')
    ax.set_ylabel('Feature')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminability_heatmap.png'), dpi=200)
    plt.close()

    # ---- SNA vs Chaos top 10 ----
    sna_chaos_col = None
    for candidate in [f'd_SNA_vs_Chaos', f'd_Chaos_vs_SNA']:
        if candidate in disc_df.columns:
            sna_chaos_col = candidate
            break
    if sna_chaos_col is not None:
        sna_chaos_df = (disc_df[['feature', sna_chaos_col, 'anova_F', 'anova_p']]
                        .rename(columns={sna_chaos_col: 'abs_cohens_d_SNA_vs_Chaos'})
                        .sort_values('abs_cohens_d_SNA_vs_Chaos', ascending=False)
                        .head(10))
        sna_chaos_df.to_csv(
            os.path.join(output_dir, 'sna_vs_chaos_top10.csv'), index=False)
    else:
        sna_chaos_df = None

    # ---- Top features per pair (plain text) ----
    with open(os.path.join(output_dir, 'top_features_per_pair.txt'), 'w',
               encoding='utf-8') as f:
        f.write("Top 5 features per regime pair, ranked by |Cohen's d|\n")
        f.write("=" * 60 + "\n\n")
        for r1, r2 in regime_pairs:
            col = f'd_{r1}_vs_{r2}'
            top5 = (disc_df[['feature', col]]
                    .dropna()
                    .sort_values(col, ascending=False)
                    .head(5))
            f.write(f"{r1} vs {r2}\n")
            f.write("-" * 30 + "\n")
            for _, row in top5.iterrows():
                f.write(f"  {row['feature']:<35s}  |d| = {row[col]:.3f}\n")
            f.write("\n")

    return {
        'discriminability': disc_df,
        'sna_vs_chaos_top10': sna_chaos_df,
        'regime_pairs': regime_pairs,
    }
