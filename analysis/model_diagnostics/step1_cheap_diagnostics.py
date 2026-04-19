"""Part A: Cheap diagnostics from cached predictions — no retraining."""
import os, sys
import numpy as np
import pandas as pd

np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
        'axes.titlesize': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
    })

import seaborn as sns
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from metric_utils import per_class_metrics, bootstrap_accuracy_ci

PROJ = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PROJ, 'results')
df = pd.read_csv(os.path.join(PROJ, '..', 'results', 'all_predictions.csv'))

CLASS_NAMES_5C = ['Limit Cycle', 'Period-2', 'Quasi-periodic', 'SNA', 'Chaos']
CLASS_NAMES_3C = ['Periodic', 'Quasi-periodic', 'Aperiodic']

MODELS = {
    'classical': {
        'SVM_comb':  'SVM (RBF)',
        'RF_comb':   'Random Forest',
        'XGB_comb':  'XGBoost',
    },
    'deep_learning': {
        '1D-CNN':    '1D-CNN',
        'LSTM':      'LSTM',
        'GRU':       'GRU',
        '2D-CNN_RP': '2D-CNN (RP)',
    },
    'ensembles': {
        'Hard-Vote-All7':       'Hard Vote',
        'Soft-Vote-All7':       'Soft Vote',
        'Soft-Vote-Classical':  'SV-Classical',
        'Soft-Vote-Deep':       'SV-Deep',
        'Stacking-LogReg':      'Stacking',
    },
}

HYBRID_5C = 'Hybrid(RF_comb+2D-CNN_RP)'
HYBRID_3C = 'Hybrid(SVM_comb+1D-CNN)'

METHOD_COLORS = {'classical': '#0C5DA5', 'deep_learning': '#FF9500',
                 'ensembles': '#00B945'}

# ========== 1. PER-CLASS METRICS ==========
print("=== Per-class metrics ===")
out_pcm = os.path.join(RESULTS, 'per_class_metrics')
f1_macro_data = {}

for nc, class_names in [(5, CLASS_NAMES_5C), (3, CLASS_NAMES_3C)]:
    true_col = f'true_{nc}c'
    for family, models in MODELS.items():
        rows = []
        for col_key, display_name in models.items():
            pred_col = f'pred_{nc}c_{col_key}'
            if pred_col not in df.columns:
                continue
            metrics = per_class_metrics(df[true_col].values, df[pred_col].values,
                                         class_names)
            for m in metrics:
                m['model'] = display_name
            rows.extend(metrics)
            macro_row = [m for m in metrics if m['class'] == 'MACRO'][0]
            f1_macro_data[(display_name, nc)] = macro_row['f1']

        if nc == 5 and family == 'ensembles':
            pred_col = f'pred_5c_{HYBRID_5C}'
            if pred_col in df.columns:
                metrics = per_class_metrics(df[true_col].values,
                                             df[pred_col].values, class_names)
                for m in metrics:
                    m['model'] = 'Hybrid'
                rows.extend(metrics)
                macro_row = [m for m in metrics if m['class'] == 'MACRO'][0]
                f1_macro_data[('Hybrid', nc)] = macro_row['f1']
        elif nc == 3 and family == 'ensembles':
            pred_col = f'pred_3c_{HYBRID_3C}'
            if pred_col in df.columns:
                metrics = per_class_metrics(df[true_col].values,
                                             df[pred_col].values, class_names)
                for m in metrics:
                    m['model'] = 'Hybrid'
                rows.extend(metrics)
                macro_row = [m for m in metrics if m['class'] == 'MACRO'][0]
                f1_macro_data[('Hybrid', nc)] = macro_row['f1']

        out_csv = os.path.join(out_pcm, f'{family}_{nc}class.csv')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  {out_csv}")

# F1-macro heatmap
all_models_ordered = []
for fam in ['classical', 'deep_learning', 'ensembles']:
    all_models_ordered.extend(MODELS[fam].values())
all_models_ordered.append('Hybrid')

matrix = np.full((len(all_models_ordered), 2), np.nan)
for i, m in enumerate(all_models_ordered):
    if (m, 5) in f1_macro_data:
        matrix[i, 0] = f1_macro_data[(m, 5)]
    if (m, 3) in f1_macro_data:
        matrix[i, 1] = f1_macro_data[(m, 3)]

fig, ax = plt.subplots(figsize=(4, 8))
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=['5-class', '3-class'], yticklabels=all_models_ordered,
            cbar_kws={'label': 'F1-macro'}, linewidths=0.5, ax=ax)
ax.set_title('F1-macro across all models')
plt.tight_layout()
plt.savefig(os.path.join(out_pcm, 'per_class_summary_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  per_class_summary_heatmap.png")

# ========== 2. SNA AND CHAOS RECALL ==========
print("\n=== SNA / Chaos recall ===")
out_sc = os.path.join(RESULTS, 'sna_chaos_focus')

for target_class, target_idx, fname in [('SNA', 3, 'sna_recall_all_models.csv'),
                                          ('Chaos', 4, 'chaos_recall_all_models.csv')]:
    mask = df['true_5c'] == target_idx
    total = mask.sum()
    rows = []
    for family, models in MODELS.items():
        for col_key, display in models.items():
            pred_col = f'pred_5c_{col_key}'
            if pred_col not in df.columns:
                continue
            correct = (df.loc[mask, pred_col] == target_idx).sum()
            rows.append({'model': display, 'family': family,
                         'recall': correct / max(total, 1),
                         'correct_count': int(correct), 'total': int(total)})
    for hybrid_col, label in [(HYBRID_5C, 'Hybrid')]:
        pred_col = f'pred_5c_{hybrid_col}'
        if pred_col in df.columns:
            correct = (df.loc[mask, pred_col] == target_idx).sum()
            rows.append({'model': label, 'family': 'ensembles',
                         'recall': correct / max(total, 1),
                         'correct_count': int(correct), 'total': int(total)})
    pd.DataFrame(rows).to_csv(os.path.join(out_sc, fname), index=False)
    print(f"  {fname}: {target_class} total={total}")

# SNA/Chaos recall bar chart
sna_df = pd.read_csv(os.path.join(out_sc, 'sna_recall_all_models.csv'))
chaos_df = pd.read_csv(os.path.join(out_sc, 'chaos_recall_all_models.csv'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for ax, rdf, title in [(ax1, sna_df, 'SNA recall (n=2)'),
                         (ax2, chaos_df, 'Chaos recall (n=4)')]:
    colors = [METHOD_COLORS.get(f, '#999') for f in rdf['family']]
    bars = ax.barh(range(len(rdf)), rdf['recall'] * 100, color=colors,
                   edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf['model'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Recall (%)')
    ax.set_xlim(0, 110)
    ax.set_title(title)
    for bar, r in zip(bars, rdf['recall']):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f'{r*100:.0f}%', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_sc, 'sna_chaos_recall_barchart.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  sna_chaos_recall_barchart.png")

# ========== 3. BOOTSTRAP 95% CI ==========
print("\n=== Bootstrap 95% CI ===")
out_cv = os.path.join(RESULTS, 'cv_robustness')
ci_rows = []

all_models_for_ci = {}
for family, models in MODELS.items():
    for col_key, display in models.items():
        all_models_for_ci[col_key] = (display, family)
all_models_for_ci[HYBRID_5C] = ('Hybrid', 'ensembles')
all_models_for_ci[HYBRID_3C] = ('Hybrid', 'ensembles')

for nc in [5, 3]:
    true_col = f'true_{nc}c'
    for col_key, (display, family) in all_models_for_ci.items():
        pred_col = f'pred_{nc}c_{col_key}'
        if pred_col not in df.columns:
            continue
        if nc == 5 and col_key == HYBRID_3C:
            continue
        if nc == 3 and col_key == HYBRID_5C:
            continue
        ci = bootstrap_accuracy_ci(df[true_col].values, df[pred_col].values)
        ci_rows.append({
            'model': display, 'family': family, 'n_classes': nc,
            'accuracy': round(ci['accuracy'] * 100, 1),
            'ci_lower': round(ci['ci_lower'] * 100, 1),
            'ci_upper': round(ci['ci_upper'] * 100, 1),
            'ci_width': round(ci['ci_width'] * 100, 1),
        })

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(os.path.join(out_cv, 'cv_accuracy_with_std.csv'), index=False)
print(f"  cv_accuracy_with_std.csv: {len(ci_df)} entries")

# ========== 4. PER-FOLD BREAKDOWN HEATMAP ==========
print("\n=== Per-fold breakdown ===")
ABBREV_5C = {0: 'LC', 1: 'P2', 2: 'QP', 3: 'SNA', 4: 'Ch'}

nc = 5
true_col = f'true_{nc}c'
true_vals = df[true_col].values
LD_vals = df['L_D'].values
LABELS_5C = {0: 'Limit Cycle', 1: 'Period-2', 2: 'Quasi-periodic',
             3: 'SNA', 4: 'Chaos'}

methods = []
correct_arr_list = []
annot_arr_list = []

for family, models in MODELS.items():
    for col_key, display in models.items():
        pred_col = f'pred_{nc}c_{col_key}'
        if pred_col not in df.columns:
            continue
        preds = df[pred_col].values
        methods.append(display)
        correct_arr_list.append((true_vals == preds).astype(int))
        annot_arr_list.append([ABBREV_5C[int(p)] for p in preds])

pred_col = f'pred_{nc}c_{HYBRID_5C}'
if pred_col in df.columns:
    preds = df[pred_col].values
    methods.append('Hybrid')
    correct_arr_list.append((true_vals == preds).astype(int))
    annot_arr_list.append([ABBREV_5C[int(p)] for p in preds])

correct_matrix = np.array(correct_arr_list).T
annot_matrix = np.array(annot_arr_list).T

ld_labels = [f'{LD_vals[i]:.3f}  {LABELS_5C[int(true_vals[i])]}'
             for i in range(len(LD_vals))]
accs = [f'{c.mean()*100:.0f}%' for c in correct_arr_list]
col_labels = [f'{m}\n({a})' for m, a in zip(methods, accs)]

cmap = mcolors.ListedColormap(['#d32f2f', '#4caf50'])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(max(8, len(methods) * 0.6), 7))
im = ax.imshow(correct_matrix, cmap=cmap, norm=norm, aspect='auto')
for i in range(correct_matrix.shape[0]):
    for j in range(correct_matrix.shape[1]):
        ax.text(j, i, annot_matrix[i, j], ha='center', va='center',
                fontsize=6, color='white', fontweight='bold')
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=6)
ax.set_yticks(range(len(ld_labels)))
ax.set_yticklabels(ld_labels, fontsize=7)
ax.set_ylabel('L/D \u2014 true regime')
ax.set_title('Per-fold correctness (5-class, green=correct, red=wrong)')
plt.tight_layout()
plt.savefig(os.path.join(out_cv, 'cv_per_fold_breakdown.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  cv_per_fold_breakdown.png")

# ========== SUMMARY ==========
n_figs = 3  # heatmap + barchart + fold breakdown
n_csvs = 8  # 6 per-class + 2 sna/chaos + 1 ci
print(f"\n=== Step 1 complete: {n_figs} figures, {n_csvs} CSVs ===")
