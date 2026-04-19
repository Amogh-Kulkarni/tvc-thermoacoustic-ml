"""
Step 4: Compare baseline / NLD2-only / combined experiments.

Produces:
    accuracy_comparison.png      grouped bar chart (5c and 3c subplots)
    per_sample_changes.csv       per-L/D correctness across experiments
    per_sample_heatmap.png       20 x 6 green/red correctness grid
    sna_chaos_focus.csv          6 aperiodic rows with all predictions
    summary.md                   narrative comparison + recommendations
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

RESULTS_DIR = 'results'
COMP_DIR = os.path.join(RESULTS_DIR, 'comparison')
os.makedirs(COMP_DIR, exist_ok=True)

EXPERIMENTS = ['experiment_1_baseline', 'experiment_2_nld2_only', 'experiment_3_combined']
EXP_SHORT = {
    'experiment_1_baseline':  'baseline',
    'experiment_2_nld2_only': 'nld2_only',
    'experiment_3_combined':  'combined',
}

EXP_COLORS = {
    'experiment_1_baseline':  '#0C5DA5',
    'experiment_2_nld2_only': '#00B945',
    'experiment_3_combined':  '#FF9500',
}

REGIME_LABELS_5C = {0: 'Limit Cycle', 1: 'Period-2', 2: 'Quasi-periodic',
                     3: 'SNA', 4: 'Chaos'}
REGIME_LABELS_3C = {0: 'Periodic', 1: 'Quasi-periodic', 2: 'Aperiodic'}


# =========================================================================
# Load everything
# =========================================================================

def load_results():
    pkl = os.path.join(RESULTS_DIR, 'all_results.pkl')
    if not os.path.exists(pkl):
        print(f"ERROR: {pkl} not found. Run step3 first.")
        sys.exit(1)
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_feature_names():
    baseline = np.load(os.path.join('..', 'results', 'features.npz'),
                        allow_pickle=True)
    nld2 = np.load(os.path.join(RESULTS_DIR, 'nld2_aligned.npz'),
                    allow_pickle=True)
    return (list(baseline['feature_names_combined']),
            list(nld2['feature_names_nld2']))


# =========================================================================
# Figure: grouped bar chart of accuracies
# =========================================================================

def plot_accuracy_comparison(all_results, save_path):
    # Get classifier names from the first experiment
    classifiers = list(all_results[EXPERIMENTS[0]].keys())
    n_clf = len(classifiers)
    n_exp = len(EXPERIMENTS)

    x = np.arange(n_clf)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [
        (axes[0], '5c_accuracy', '5-class accuracy'),
        (axes[1], '3c_accuracy', '3-class accuracy'),
    ]:
        for i, exp in enumerate(EXPERIMENTS):
            vals = [all_results[exp][c][key] * 100 for c in classifiers]
            offset = (i - (n_exp - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                           label=EXP_SHORT[exp], color=EXP_COLORS[exp],
                           edgecolor='black', linewidth=0.4)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.set_ylim(0, 110)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# =========================================================================
# Per-sample changes (use Random Forest as representative classifier)
# =========================================================================

def build_per_sample_changes(all_results):
    # Use Random Forest as the representative classifier; fall back to first
    rep = 'Random_Forest' if 'Random_Forest' in all_results[EXPERIMENTS[0]] else \
          list(all_results[EXPERIMENTS[0]].keys())[0]

    # Grab LD_test and true labels from the first experiment
    first = all_results[EXPERIMENTS[0]][rep]
    ld = first['LD_test']
    true_5c = first['5c_y_true']
    true_3c = first['3c_y_true']

    rows = []
    for i in range(len(ld)):
        row = {
            'L_D': float(ld[i]),
            'regime_5c': REGIME_LABELS_5C[int(true_5c[i])],
            'regime_3c': REGIME_LABELS_3C[int(true_3c[i])],
        }
        for exp in EXPERIMENTS:
            r = all_results[exp][rep]
            row[f'{EXP_SHORT[exp]}_5c_pred'] = REGIME_LABELS_5C[int(r['5c_y_pred'][i])]
            row[f'{EXP_SHORT[exp]}_5c_correct'] = bool(r['5c_y_true'][i] == r['5c_y_pred'][i])
            row[f'{EXP_SHORT[exp]}_3c_pred'] = REGIME_LABELS_3C[int(r['3c_y_pred'][i])]
            row[f'{EXP_SHORT[exp]}_3c_correct'] = bool(r['3c_y_true'][i] == r['3c_y_pred'][i])

        # Improvement / regression indicators (combined vs baseline)
        row['improvement_5c'] = (row['combined_5c_correct'] and not row['baseline_5c_correct'])
        row['regression_5c']  = (not row['combined_5c_correct'] and row['baseline_5c_correct'])
        row['improvement_3c'] = (row['combined_3c_correct'] and not row['baseline_3c_correct'])
        row['regression_3c']  = (not row['combined_3c_correct'] and row['baseline_3c_correct'])
        rows.append(row)

    return pd.DataFrame(rows).sort_values('L_D').reset_index(drop=True)


def plot_per_sample_heatmap(per_sample_df, save_path):
    exps = ['baseline', 'nld2_only', 'combined']
    cols_5c = [f'{e}_5c_correct' for e in exps]
    cols_3c = [f'{e}_3c_correct' for e in exps]
    all_cols = cols_5c + cols_3c
    col_labels = [f'{e} 5c' for e in exps] + [f'{e} 3c' for e in exps]

    data = per_sample_df[all_cols].astype(int).values  # 20 x 6
    fig_h = max(6, 0.3 * len(per_sample_df))
    fig, ax = plt.subplots(figsize=(8, fig_h))

    cmap = plt.cm.RdYlGn
    ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)

    ylabels = [f"L/D={row['L_D']:.3f}  {row['regime_5c']}"
               for _, row in per_sample_df.iterrows()]
    ax.set_yticks(range(len(per_sample_df)))
    ax.set_yticklabels(ylabels, fontsize=8)

    # Annotate each cell with the predicted regime short form
    pred_cols_5c = [f'{e}_5c_pred' for e in exps]
    pred_cols_3c = [f'{e}_3c_pred' for e in exps]
    label_matrix = []
    for _, row in per_sample_df.iterrows():
        rlabels = []
        for c in pred_cols_5c + pred_cols_3c:
            pred = row[c]
            short = {'Limit Cycle': 'LC', 'Period-2': 'P2',
                     'Quasi-periodic': 'QP', 'SNA': 'SNA', 'Chaos': 'Ch',
                     'Periodic': 'Per', 'Aperiodic': 'Ap'}.get(pred, pred[:4])
            rlabels.append(short)
        label_matrix.append(rlabels)

    for i, labels in enumerate(label_matrix):
        for j, lbl in enumerate(labels):
            ax.text(j, i, lbl, ha='center', va='center',
                    fontsize=7, fontweight='bold')

    ax.set_title('Per-sample correctness (Random Forest)\n'
                 'green = correct, red = wrong; annotation = predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# =========================================================================
# SNA / Chaos focus CSV
# =========================================================================

def build_sna_chaos_focus(all_results):
    first_clf = list(all_results[EXPERIMENTS[0]].keys())[0]
    ld = all_results[EXPERIMENTS[0]][first_clf]['LD_test']
    true_5c = all_results[EXPERIMENTS[0]][first_clf]['5c_y_true']

    # Indices of aperiodic samples (label 3 or 4 in 5-class)
    aperi_mask = np.isin(true_5c, [3, 4])
    idx = np.where(aperi_mask)[0]

    rows = []
    for i in idx:
        row = {
            'L_D': float(ld[i]),
            'true_5c': REGIME_LABELS_5C[int(true_5c[i])],
        }
        for exp in EXPERIMENTS:
            for clf_name, clf_res in all_results[exp].items():
                col = f'{EXP_SHORT[exp]}_{clf_name}_5c'
                row[col] = REGIME_LABELS_5C[int(clf_res['5c_y_pred'][i])]
        rows.append(row)
    return pd.DataFrame(rows).sort_values('L_D').reset_index(drop=True)


# =========================================================================
# Markdown report
# =========================================================================

def write_summary(all_results, per_sample_df, sna_chaos_df,
                   feature_names_baseline, feature_names_nld2, save_path):
    lines = []
    lines.append("# NLD2 Integration Experiment Report\n")

    # ---- Headline accuracies ----
    lines.append("## 1. Headline accuracies (leave-one-$L/D$-out CV)\n")
    classifiers = list(all_results[EXPERIMENTS[0]].keys())
    header = "| Classifier | Experiment | 5-class | 3-class |"
    sep = "|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for clf in classifiers:
        for exp in EXPERIMENTS:
            r = all_results[exp][clf]
            lines.append(f"| {clf} | {EXP_SHORT[exp]} | "
                         f"{r['5c_accuracy']*100:.1f}% | {r['3c_accuracy']*100:.1f}% |")
    lines.append("")

    # ---- Comparison baseline -> combined ----
    lines.append("## 2. Baseline vs combined (delta)\n")
    lines.append("| Classifier | 5c baseline | 5c combined | Δ5c | 3c baseline | 3c combined | Δ3c |")
    lines.append("|---|---|---|---|---|---|---|")
    for clf in classifiers:
        b5 = all_results['experiment_1_baseline'][clf]['5c_accuracy']
        c5 = all_results['experiment_3_combined'][clf]['5c_accuracy']
        b3 = all_results['experiment_1_baseline'][clf]['3c_accuracy']
        c3 = all_results['experiment_3_combined'][clf]['3c_accuracy']
        d5 = (c5 - b5) * 100
        d3 = (c3 - b3) * 100
        arrow5 = '+' if d5 > 0 else ''
        arrow3 = '+' if d3 > 0 else ''
        lines.append(f"| {clf} | {b5*100:.1f}% | {c5*100:.1f}% | {arrow5}{d5:.1f} pp "
                     f"| {b3*100:.1f}% | {c3*100:.1f}% | {arrow3}{d3:.1f} pp |")
    lines.append("")

    # ---- SNA recall analysis ----
    lines.append("## 3. SNA recall (5-class)\n")
    lines.append(
        "SNA recall is the fraction of true SNA recordings predicted correctly "
        "as SNA. In the main pipeline this was 0 out of 2 (no model got an SNA "
        "right on the 5-class task). Here is how each experiment performs:\n\n"
    )
    lines.append("| Classifier | Experiment | SNA hits (of 2) | Aperiodic hits (SNA+Chaos, of 6) |")
    lines.append("|---|---|---|---|")
    for clf in classifiers:
        for exp in EXPERIMENTS:
            r = all_results[exp][clf]
            true5 = r['5c_y_true']
            pred5 = r['5c_y_pred']
            sna_mask = (true5 == 3)
            sna_hits = int(((true5 == 3) & (pred5 == 3)).sum())
            aperi_mask = np.isin(true5, [3, 4])
            aperi_hits = int((pred5[aperi_mask] == true5[aperi_mask]).sum())
            lines.append(f"| {clf} | {EXP_SHORT[exp]} | {sna_hits}/{int(sna_mask.sum())} "
                         f"| {aperi_hits}/{int(aperi_mask.sum())} |")
    lines.append("")

    # ---- Boundary samples ----
    lines.append("## 4. Boundary samples (L/D = 2.0, 2.125, 2.25)\n")
    lines.append(
        "These three recordings sit on the SNA/Chaos boundary and were "
        "misclassified by every method in the main ML run. Did NLD2 features help?\n\n"
    )
    boundary = per_sample_df[per_sample_df['L_D'].round(4).isin([2.0, 2.125, 2.25])]
    if len(boundary) == 0:
        lines.append("_no boundary samples found in the dataset_\n\n")
    else:
        lines.append("| L/D | True 5c | baseline 5c | nld2_only 5c | combined 5c |")
        lines.append("|---|---|---|---|---|")
        for _, row in boundary.iterrows():
            lines.append(
                f"| {row['L_D']:.3f} | {row['regime_5c']} | "
                f"{row['baseline_5c_pred']} "
                f"({'✓' if row['baseline_5c_correct'] else '✗'}) | "
                f"{row['nld2_only_5c_pred']} "
                f"({'✓' if row['nld2_only_5c_correct'] else '✗'}) | "
                f"{row['combined_5c_pred']} "
                f"({'✓' if row['combined_5c_correct'] else '✗'}) |")
        lines.append("")

    # ---- Feature importance: NLD2 features in combined experiment ----
    lines.append("## 5. NLD2 features in the combined experiment (RF top-20)\n")
    imp_csv = os.path.join(RESULTS_DIR, 'experiment_3_combined', 'feature_importance_5c.csv')
    if os.path.exists(imp_csv):
        imp = pd.read_csv(imp_csv)
        top20 = imp.head(20).reset_index(drop=True)
        nld2_in_top20 = [f for f in top20['feature'] if str(f).startswith('nld2_')]
        lines.append(
            f"- Top-20 features in the combined experiment: {len(top20)}\n"
            f"- Of those, **{len(nld2_in_top20)} are NLD2 features**\n\n"
        )
        if nld2_in_top20:
            lines.append("NLD2 features that made the top 20:\n\n")
            for f in nld2_in_top20:
                rank = int(top20[top20['feature'] == f].index[0]) + 1
                val = float(top20[top20['feature'] == f]['importance'].iloc[0])
                lines.append(f"- rank {rank:2d}: `{f}` (importance = {val:.4f})")
            lines.append("")
    else:
        lines.append("_(feature importance CSV not found)_\n\n")

    # ---- SNA/Chaos focus table ----
    lines.append("## 6. All model predictions on the 6 aperiodic samples\n")
    lines.append("Rows = aperiodic recordings; columns = {experiment}_{classifier}.\n\n")
    header_cols = [c for c in sna_chaos_df.columns if c not in ['L_D', 'true_5c']]
    lines.append("| L/D | True | " + " | ".join(header_cols) + " |")
    lines.append("|---|---|" + "|".join(['---'] * len(header_cols)) + "|")
    for _, row in sna_chaos_df.iterrows():
        cells = [f"{row['L_D']:.3f}", row['true_5c']]
        for col in header_cols:
            val = row[col]
            mark = '✓' if val == row['true_5c'] else '✗'
            cells.append(f"{val} {mark}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Honest assessment ----
    lines.append("## 7. Honest assessment\n")
    # Compute deltas to drive the conclusions
    baseline_5c = [all_results['experiment_1_baseline'][c]['5c_accuracy'] for c in classifiers]
    combined_5c = [all_results['experiment_3_combined'][c]['5c_accuracy'] for c in classifiers]
    best_baseline_5c = max(baseline_5c)
    best_combined_5c = max(combined_5c)
    delta_best_5c = (best_combined_5c - best_baseline_5c) * 100

    baseline_3c = [all_results['experiment_1_baseline'][c]['3c_accuracy'] for c in classifiers]
    combined_3c = [all_results['experiment_3_combined'][c]['3c_accuracy'] for c in classifiers]
    best_baseline_3c = max(baseline_3c)
    best_combined_3c = max(combined_3c)
    delta_best_3c = (best_combined_3c - best_baseline_3c) * 100

    lines.append(
        f"- **Best 5-class accuracy:** baseline {best_baseline_5c*100:.1f}%, "
        f"combined {best_combined_5c*100:.1f}%, delta = {delta_best_5c:+.1f} pp\n"
        f"- **Best 3-class accuracy:** baseline {best_baseline_3c*100:.1f}%, "
        f"combined {best_combined_3c*100:.1f}%, delta = {delta_best_3c:+.1f} pp\n\n"
    )

    # SNA recall delta
    sna_baseline = sum(
        int(((r['5c_y_true'] == 3) & (r['5c_y_pred'] == 3)).sum())
        for r in all_results['experiment_1_baseline'].values()
    )
    sna_combined = sum(
        int(((r['5c_y_true'] == 3) & (r['5c_y_pred'] == 3)).sum())
        for r in all_results['experiment_3_combined'].values()
    )
    total_sna_tests = len(classifiers) * 2  # 2 SNA samples per classifier

    lines.append(
        f"- **SNA recall (across all classifiers):** baseline {sna_baseline}/{total_sna_tests}, "
        f"combined {sna_combined}/{total_sna_tests}\n\n"
    )

    if delta_best_5c > 5:
        verdict_5c = ("The combined feature set gives a meaningful gain on the 5-class "
                       "problem. Worth including NLD2 features in the final report.")
    elif delta_best_5c > 0:
        verdict_5c = ("The combined feature set gives a small but real improvement on "
                       "the 5-class problem. Worth mentioning but not headline-worthy.")
    elif delta_best_5c > -3:
        verdict_5c = ("The combined feature set gives no meaningful gain. The baseline "
                       "features already captured most of the relevant information.")
    else:
        verdict_5c = ("The combined feature set *regressed* relative to baseline. "
                       "Probably a curse-of-dimensionality effect with 20 training samples.")

    lines.append(f"**Verdict (5-class):** {verdict_5c}\n\n")

    if sna_combined > sna_baseline:
        sna_verdict = (f"SNA recall improved ({sna_baseline} -> {sna_combined} across all classifiers). "
                        "NLD2 features offered real information about the SNA boundary.")
    elif sna_combined == sna_baseline:
        sna_verdict = ("SNA recall did not change. With only 2 SNA samples this is an "
                        "extremely noisy signal and no conclusion is robust.")
    else:
        sna_verdict = ("SNA recall got worse. Fluctuation on 2 samples; not statistically meaningful.")

    lines.append(f"**Verdict (SNA boundary):** {sna_verdict}\n\n")

    # ---- Recommendation ----
    lines.append("## 8. Recommendation for the final report\n")
    if delta_best_5c > 2:
        rec = (
            f"Include the combined feature set ({best_combined_5c*100:.1f}% 5-class) "
            "as the primary ML result. Mention the baseline as an ablation showing "
            "that NLD2 features add measurable value beyond the original 93 features."
        )
    else:
        rec = (
            "Stick with the original combined feature set from main_classical_ml.py "
            f"({best_baseline_5c*100:.1f}% 5-class, up to {best_baseline_3c*100:.0f}% 3-class). "
            "The NLD2 integration does not provide a meaningful gain on this 20-recording "
            "dataset. Report the NLD2 analysis as a separate analytical validation of the "
            "regime characterizations, not as an ML feature source."
        )
    lines.append(rec + "\n\n")
    lines.append(
        "**Caveat on all conclusions:** with 20 recordings and only 2 SNA samples, "
        "any accuracy delta below roughly 5 percentage points is within the "
        "resampling noise of a single classifier. Differences at the SNA boundary "
        "are particularly unreliable and should be treated as anecdotal.\n"
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# =========================================================================
# Main
# =========================================================================

def main():
    all_results = load_results()
    feature_names_baseline, feature_names_nld2 = load_feature_names()

    # 1. Grouped bar chart
    plot_accuracy_comparison(all_results,
                               os.path.join(COMP_DIR, 'accuracy_comparison.png'))
    print("Wrote accuracy_comparison.png")

    # 2. Per-sample changes CSV + heatmap
    per_sample = build_per_sample_changes(all_results)
    per_sample.to_csv(os.path.join(COMP_DIR, 'per_sample_changes.csv'), index=False)
    print(f"Wrote per_sample_changes.csv ({len(per_sample)} rows)")

    plot_per_sample_heatmap(per_sample,
                              os.path.join(COMP_DIR, 'per_sample_heatmap.png'))
    print("Wrote per_sample_heatmap.png")

    # 3. SNA/Chaos focus
    sna_chaos = build_sna_chaos_focus(all_results)
    sna_chaos.to_csv(os.path.join(COMP_DIR, 'sna_chaos_focus.csv'), index=False)
    print(f"Wrote sna_chaos_focus.csv ({len(sna_chaos)} rows)")

    # 4. Summary markdown
    write_summary(all_results, per_sample, sna_chaos,
                   feature_names_baseline, feature_names_nld2,
                   os.path.join(COMP_DIR, 'summary.md'))
    print("Wrote summary.md")

    print(f"\nAll comparison artifacts in {COMP_DIR}/")


if __name__ == "__main__":
    main()
