"""
Step 1: Pairwise Cohen's d + ANOVA diagnostic on NLD2 features.

Reads the NLD2 per-recording table (already computed by
nonlinear_dynamics_analysis_2) and produces the diagnostic artifacts
in results/pairwise_diagnostic/.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import pandas as pd  # noqa: E402
from pairwise_diagnostic import analyze_discriminability  # noqa: E402

np.random.seed(42)

NLD2_CSV = os.path.join(
    '..', 'nonlinear_dynamics_analysis_2', 'results', 'summary_tables',
    'features_mean.csv')
OUTPUT_DIR = os.path.join('results', 'pairwise_diagnostic')


def main():
    if not os.path.exists(NLD2_CSV):
        print(f"ERROR: {NLD2_CSV} not found.")
        print("Run nonlinear_dynamics_analysis_2/main_analysis.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(NLD2_CSV)
    print(f"Loaded {len(df)} recordings, {len(df.columns)} columns")
    print(f"  columns: {list(df.columns)}\n")

    result = analyze_discriminability(
        df,
        regime_col='regime_name',
        exclude_cols=['L_D', 'filename', 'regime_label'],
        output_dir=OUTPUT_DIR,
    )

    print(f"\nSaved diagnostic artifacts to {OUTPUT_DIR}/")

    # Print top 5 per pair (nicer reading than the .txt file)
    disc = result['discriminability']
    pairs = result['regime_pairs']
    print("\n" + "=" * 70)
    print("Top 5 features per regime pair, ranked by |Cohen's d|")
    print("=" * 70)
    for r1, r2 in pairs:
        col = f'd_{r1}_vs_{r2}'
        top5 = (disc[['feature', col]]
                .dropna()
                .sort_values(col, ascending=False)
                .head(5))
        print(f"\n{r1} vs {r2}:")
        for _, row in top5.iterrows():
            print(f"  {row['feature']:<35s}  |d| = {row[col]:.3f}")

    # Call out SNA vs Chaos specifically
    sna_chaos_col = None
    for candidate in ['d_SNA_vs_Chaos', 'd_Chaos_vs_SNA']:
        if candidate in disc.columns:
            sna_chaos_col = candidate
            break
    if sna_chaos_col is not None:
        print("\n" + "=" * 70)
        print("*** SNA vs CHAOS: top 5 discriminating features ***")
        print("=" * 70)
        top5 = (disc[['feature', sna_chaos_col, 'anova_F']]
                .dropna(subset=[sna_chaos_col])
                .sort_values(sna_chaos_col, ascending=False)
                .head(5))
        for _, row in top5.iterrows():
            print(f"  {row['feature']:<35s}  |d| = {row[sna_chaos_col]:.3f}"
                  f"  (ANOVA F = {row['anova_F']:.1f})")

    print(f"\nDone. See {OUTPUT_DIR}/ for the full outputs.")


if __name__ == "__main__":
    main()
