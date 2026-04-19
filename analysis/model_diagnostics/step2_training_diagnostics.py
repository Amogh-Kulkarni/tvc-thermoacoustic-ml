"""Part B: DL training curves and seed stability — requires retraining."""
import os, sys, time
import numpy as np

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

import torch
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJ, 'src'))
sys.path.insert(0, os.path.join(PROJ, '..', 'src'))
sys.path.insert(0, os.path.join(PROJ, '..'))

from training_with_history import (
    run_cnn1d_cv_with_history, run_rnn_cv_with_history,
    run_cnn2d_cv_with_history, DEVICE
)
from data_loading import load_all_data

RESULTS = os.path.join(PROJ, 'results')
TC_DIR = os.path.join(RESULTS, 'training_curves')
PLOT_DIR = os.path.join(TC_DIR, 'plots')
SEED_DIR = os.path.join(RESULTS, 'seed_stability')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SEED_DIR, exist_ok=True)

DATA_DIR = os.path.join(PROJ, '..', 'data')
print(f"Device: {DEVICE}")
print(f"Loading data from {DATA_DIR}...")
dataset = load_all_data(DATA_DIR)
print(f"Loaded {len(dataset)} recordings")

MODEL_RUNNERS = {
    '1DCNN': lambda ds, nc, seed, hist: run_cnn1d_cv_with_history(
        ds, nc, seed=seed, capture_history=hist),
    'LSTM': lambda ds, nc, seed, hist: run_rnn_cv_with_history(
        ds, nc, model_type='LSTM', seed=seed, capture_history=hist),
    'GRU': lambda ds, nc, seed, hist: run_rnn_cv_with_history(
        ds, nc, model_type='GRU', seed=seed, capture_history=hist),
    '2DCNN_RP': lambda ds, nc, seed, hist: run_cnn2d_cv_with_history(
        ds, nc, seed=seed, capture_history=hist),
}

ORIGINAL_ACC = {
    ('1DCNN', 5): 75.0, ('LSTM', 5): 55.0, ('GRU', 5): 55.0,
    ('2DCNN_RP', 5): 80.0,
    ('1DCNN', 3): 90.0, ('LSTM', 3): 85.0, ('GRU', 3): 85.0,
    ('2DCNN_RP', 3): 85.0,
}

# ========== PART 1: TRAINING CURVES ==========
print("\n" + "=" * 70)
print("  TRAINING CURVES (8 runs: 4 models x 2 class schemes)")
print("=" * 70)

history_results = {}

for model_name, runner in MODEL_RUNNERS.items():
    for nc in [5, 3]:
        key = f'{model_name}_{nc}c'
        print(f"\n--- {model_name} ({nc}-class) ---")
        t0 = time.time()
        try:
            result = runner(dataset, nc, seed=42, hist=True)
            elapsed = time.time() - t0
            acc = result['accuracy'] * 100
            orig = ORIGINAL_ACC.get((model_name, nc), 0)
            delta = acc - orig
            print(f"  Accuracy: {acc:.1f}% (original: {orig:.1f}%, "
                  f"delta: {delta:+.1f}%) [{elapsed:.1f}s]")
            if abs(delta) > 20:
                print(f"  WARNING: large deviation from original ({delta:+.1f}%)")
            history_results[key] = result

            # Save history as npz
            max_len = max(len(h['train_loss'])
                          for h in result['fold_histories'])
            def pad(arr, length):
                out = np.full(length, np.nan)
                out[:len(arr)] = arr
                return out

            histories = result['fold_histories']
            np.savez(os.path.join(TC_DIR, f'history_{key}.npz'),
                     train_loss=np.array([pad(h['train_loss'], max_len)
                                           for h in histories]),
                     val_loss=np.array([pad(h['val_loss'], max_len)
                                         for h in histories]),
                     train_acc=np.array([pad(h['train_acc'], max_len)
                                          for h in histories]),
                     val_acc=np.array([pad(h['val_acc'], max_len)
                                        for h in histories]),
                     best_epochs=result['best_epochs'],
                     accuracy=result['accuracy'])
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}")

# Plot training curves
print("\n--- Plotting training curves ---")
for key, result in history_results.items():
    histories = result['fold_histories']
    max_len = max(len(h['train_loss']) for h in histories)

    def pad(arr, length):
        out = np.full(length, np.nan)
        out[:len(arr)] = arr
        return out

    tl = np.array([pad(h['train_loss'], max_len) for h in histories])
    vl = np.array([pad(h['val_loss'], max_len) for h in histories])
    ta = np.array([pad(h['train_acc'], max_len) for h in histories])
    va = np.array([pad(h['val_acc'], max_len) for h in histories])

    mean_tl = np.nanmean(tl, axis=0)
    std_tl = np.nanstd(tl, axis=0)
    mean_vl = np.nanmean(vl, axis=0)
    std_vl = np.nanstd(vl, axis=0)
    mean_ta = np.nanmean(ta, axis=0)
    std_ta = np.nanstd(ta, axis=0)
    mean_va = np.nanmean(va, axis=0)
    std_va = np.nanstd(va, axis=0)

    epochs = np.arange(max_len)
    mean_best = np.mean(result['best_epochs'])
    model_name = key.replace('_5c', '').replace('_3c', '')
    nc = 5 if '5c' in key else 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax1.plot(epochs, mean_tl, color='#0C5DA5', label='Train', linewidth=1.2)
    ax1.fill_between(epochs, mean_tl - std_tl, mean_tl + std_tl,
                      color='#0C5DA5', alpha=0.2)
    ax1.plot(epochs, mean_vl, color='#FF2C00', label='Validation', linewidth=1.2)
    ax1.fill_between(epochs, mean_vl - std_vl, mean_vl + std_vl,
                      color='#FF2C00', alpha=0.2)
    ax1.axvline(mean_best, color='grey', linestyle='--', linewidth=0.5,
                label=f'Mean best epoch ({mean_best:.0f})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-entropy loss')
    ax1.set_title(f'{model_name} ({nc}-class): Loss')
    ax1.legend(frameon=False, fontsize=7)

    ax2.plot(epochs, mean_ta * 100, color='#0C5DA5', label='Train', linewidth=1.2)
    ax2.fill_between(epochs, (mean_ta - std_ta) * 100,
                      (mean_ta + std_ta) * 100, color='#0C5DA5', alpha=0.2)
    ax2.plot(epochs, mean_va * 100, color='#FF2C00', label='Validation', linewidth=1.2)
    ax2.fill_between(epochs, (mean_va - std_va) * 100,
                      (mean_va + std_va) * 100, color='#FF2C00', alpha=0.2)
    ax2.axvline(mean_best, color='grey', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} ({nc}-class): Accuracy')
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'training_curves_{key}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {key}")

# Overfitting check
print("\n--- Overfitting check ---")
overfit_data = []
for key, result in history_results.items():
    histories = result['fold_histories']
    train_accs_at_best = []
    val_accs_at_best = []
    for h, be in zip(histories, result['best_epochs']):
        if be < len(h['train_acc']):
            train_accs_at_best.append(h['train_acc'][be])
            val_accs_at_best.append(h['val_acc'][be])
    if train_accs_at_best:
        overfit_data.append({
            'model': key,
            'train_acc': np.mean(train_accs_at_best) * 100,
            'val_acc': np.mean(val_accs_at_best) * 100,
            'gap': (np.mean(train_accs_at_best) - np.mean(val_accs_at_best)) * 100,
        })

if overfit_data:
    odf = pd.DataFrame(overfit_data)
    x = np.arange(len(odf))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, odf['train_acc'], w, label='Train', color='#0C5DA5',
           edgecolor='black', linewidth=0.3)
    ax.bar(x + w / 2, odf['val_acc'], w, label='Validation', color='#FF2C00',
           edgecolor='black', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(odf['model'], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy at best epoch (%)')
    ax.set_title('Train vs Validation accuracy at best epoch')
    ax.legend(frameon=False)
    for i, gap in enumerate(odf['gap']):
        ax.text(i, max(odf['train_acc'].iloc[i], odf['val_acc'].iloc[i]) + 1,
                f'gap={gap:.1f}%', ha='center', fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'overfitting_check.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  overfitting_check.png")

# ========== PART 2: SEED STABILITY ==========
print("\n" + "=" * 70)
print("  SEED STABILITY (4 models x 2 classes x 5 seeds)")
print("=" * 70)

SEEDS = [42, 7, 13, 100, 2024]
seed_rows = []

for model_name, runner in MODEL_RUNNERS.items():
    for nc in [5, 3]:
        print(f"\n--- {model_name} ({nc}-class) ---")
        accs = []
        for seed in SEEDS:
            t0 = time.time()
            try:
                result = runner(dataset, nc, seed=seed, hist=False)
                acc = result['accuracy'] * 100
                accs.append(acc)
                print(f"  seed={seed:5d}: {acc:.1f}% [{time.time()-t0:.1f}s]")
            except Exception as e:
                print(f"  seed={seed:5d}: FAILED ({e})")
                accs.append(np.nan)

        mean_acc = np.nanmean(accs)
        std_acc = np.nanstd(accs)
        print(f"  Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%")

        seed_rows.append({
            'model': model_name, 'n_classes': nc,
            'mean_accuracy': round(mean_acc, 1),
            'std_accuracy': round(std_acc, 1),
            **{f'seed_{s}': a for s, a in zip(SEEDS, accs)},
        })

seed_df = pd.DataFrame(seed_rows)
for nc in [5, 3]:
    sub = seed_df[seed_df['n_classes'] == nc]
    sub.to_csv(os.path.join(SEED_DIR, f'seed_stability_{nc}class.csv'), index=False)

# Seed stability barchart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, nc in zip(axes, [5, 3]):
    sub = seed_df[seed_df['n_classes'] == nc]
    x = np.arange(len(sub))
    ax.bar(x, sub['mean_accuracy'], yerr=sub['std_accuracy'],
           color='#0C5DA5', edgecolor='black', linewidth=0.3,
           capsize=4, error_kw={'linewidth': 1})
    ax.set_xticks(x)
    ax.set_xticklabels(sub['model'], fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{nc}-class: seed stability (mean +/- std)')
    ax.set_ylim(0, 105)
    for i, (m, s) in enumerate(zip(sub['mean_accuracy'], sub['std_accuracy'])):
        ax.text(i, m + s + 2, f'{m:.1f}\n\u00b1{s:.1f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(SEED_DIR, 'seed_stability_barchart.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("\n  seed_stability_barchart.png")

# ========== SUMMARY REPORT ==========
print("\n--- Generating summary report ---")

report_lines = ["# Model Diagnostics Report\n"]

# A. Headline table
report_lines.append("## A. Headline numbers\n")
report_lines.append("| Model | nc | Accuracy | F1-macro | SNA recall | Chaos recall "
                     "| 95% CI | Seed std |\n")
report_lines.append("|---|---|---|---|---|---|---|---|\n")

# Load cheap diagnostics
ci_csv = os.path.join(RESULTS, 'cv_robustness', 'cv_accuracy_with_std.csv')
sna_csv = os.path.join(RESULTS, 'sna_chaos_focus', 'sna_recall_all_models.csv')
chaos_csv = os.path.join(RESULTS, 'sna_chaos_focus', 'chaos_recall_all_models.csv')

ci_df = pd.DataFrame()
sna_recall = {}
chaos_recall = {}

if os.path.exists(ci_csv):
    ci_df = pd.read_csv(ci_csv)
if os.path.exists(sna_csv):
    for _, row in pd.read_csv(sna_csv).iterrows():
        sna_recall[row['model']] = f"{row['recall']*100:.0f}%"
if os.path.exists(chaos_csv):
    for _, row in pd.read_csv(chaos_csv).iterrows():
        chaos_recall[row['model']] = f"{row['recall']*100:.0f}%"

pcm_dir = os.path.join(RESULTS, 'per_class_metrics')
f1_data = {}
for fname in os.listdir(pcm_dir):
    if fname.endswith('.csv'):
        tmp = pd.read_csv(os.path.join(pcm_dir, fname))
        macro = tmp[tmp['class'] == 'MACRO']
        nc_val = 5 if '5class' in fname else 3
        for _, row in macro.iterrows():
            f1_data[(row['model'], nc_val)] = f"{row['f1']:.2f}"

DL_MODEL_MAP = {'1DCNN': '1D-CNN', 'LSTM': 'LSTM', 'GRU': 'GRU',
                '2DCNN_RP': '2D-CNN (RP)'}

for _, srow in seed_df.iterrows():
    mn = srow['model']
    nc = int(srow['n_classes'])
    display = DL_MODEL_MAP.get(mn, mn)
    acc = f"{srow['mean_accuracy']:.1f}%"
    f1 = f1_data.get((display, nc), '-')
    sna = sna_recall.get(display, '-') if nc == 5 else '-'
    ch = chaos_recall.get(display, '-') if nc == 5 else '-'
    ci_row = ci_df[(ci_df['model'] == display) & (ci_df['n_classes'] == nc)]
    ci_str = '-'
    if len(ci_row) > 0:
        ci_str = f"[{ci_row.iloc[0]['ci_lower']:.0f}%, {ci_row.iloc[0]['ci_upper']:.0f}%]"
    seed_std = f"\u00b1{srow['std_accuracy']:.1f}%"
    report_lines.append(f"| {display} | {nc} | {acc} | {f1} | {sna} | {ch} "
                        f"| {ci_str} | {seed_std} |\n")

# B/C/D/E sections
report_lines.append("\n## B. Per-class observations\n\n")
report_lines.append("- **Universally easy**: Quasi-periodic and Chaos (L/D >= 2.375) "
                     "are correctly classified by nearly all models.\n")
report_lines.append("- **Universally hard**: SNA recall is 0 for most models "
                     "(only 2 SNA samples at L/D=2.0 and 2.125, both misclassified "
                     "as Chaos).\n")
report_lines.append("- **Period-2** (n=3): noisy per-class metrics due to small support.\n")

report_lines.append("\n## C. Training behavior summary\n\n")
for key in sorted(history_results.keys()):
    result = history_results[key]
    mn = key.replace('_5c', '').replace('_3c', '')
    nc = 5 if '5c' in key else 3
    mean_best = np.mean(result['best_epochs'])
    acc = result['accuracy'] * 100
    orig = ORIGINAL_ACC.get((mn, nc), 0)
    delta = acc - orig

    orow = [o for o in overfit_data if o['model'] == key]
    gap_str = f"{orow[0]['gap']:.1f}%" if orow else "N/A"

    seed_row = seed_df[(seed_df['model'] == mn) & (seed_df['n_classes'] == nc)]
    std_str = f"{seed_row.iloc[0]['std_accuracy']:.1f}%" if len(seed_row) > 0 else "N/A"

    report_lines.append(f"### {mn} ({nc}-class)\n")
    report_lines.append(f"- Accuracy: {acc:.1f}% (original: {orig:.1f}%, "
                        f"delta: {delta:+.1f}%)\n")
    report_lines.append(f"- Mean best epoch: {mean_best:.0f}\n")
    report_lines.append(f"- Train-val gap: {gap_str}\n")
    report_lines.append(f"- Seed stability std: {std_str}\n\n")

report_lines.append("\n## D. Recommendations for PPT\n\n")
report_lines.append("- Present 2D-CNN (RP) and 1D-CNN as strongest DL results "
                     "(highest accuracy, most distinct architectures).\n")
report_lines.append("- Report seed stability alongside accuracy to show "
                     "confidence in results.\n")
report_lines.append("- Frame SNA recall = 0 as a dataset limitation (n=2), "
                     "not a model failure.\n")

report_lines.append("\n## E. Caveats\n\n")
report_lines.append("- 20 recordings is the fundamental constraint.\n")
report_lines.append("- Bootstrap CIs on 20 samples are necessarily wide.\n")
report_lines.append("- Seed stability matters more than usual at this sample size.\n")
report_lines.append("- Per-class metrics for SNA (n=2) and Period-2 (n=3) are "
                     "inherently noisy.\n")

with open(os.path.join(RESULTS, 'summary_report.md'), 'w') as f:
    f.writelines(report_lines)
print("  summary_report.md")

print("\n=== Step 2 complete ===")
