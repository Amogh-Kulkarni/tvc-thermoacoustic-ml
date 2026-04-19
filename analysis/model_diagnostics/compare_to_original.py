"""Compare step 2 training diagnostics against original DL results."""
import os, sys
import numpy as np
import pandas as pd

np.random.seed(42)

PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, 'results', 'comparison_to_original')
os.makedirs(OUT, exist_ok=True)

# --- Check prerequisites ---
seed_5c = os.path.join(PROJ, 'results', 'seed_stability', 'seed_stability_5class.csv')
seed_3c = os.path.join(PROJ, 'results', 'seed_stability', 'seed_stability_3class.csv')
pred_csv = os.path.join(PROJ, '..', 'results', 'all_predictions.csv')

for f in [seed_5c, seed_3c, pred_csv]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found. Run step 2 first.")
        sys.exit(1)

seed_5 = pd.read_csv(seed_5c)
seed_3 = pd.read_csv(seed_3c)
pred_df = pd.read_csv(pred_csv)

MODEL_COL_MAP = {
    '1DCNN': '1D-CNN',
    'LSTM': 'LSTM',
    'GRU': 'GRU',
    '2DCNN_RP': '2D-CNN_RP',
}

# --- Level 1 & 2: Accuracy consistency + seed stability ---
rows = []
for nc in [5, 3]:
    seed_df = seed_5 if nc == 5 else seed_3
    true_col = f'true_{nc}c'

    for _, srow in seed_df.iterrows():
        model_key = srow['model']
        pred_col_name = MODEL_COL_MAP[model_key]
        pred_col = f'pred_{nc}c_{pred_col_name}'

        if pred_col not in pred_df.columns:
            print(f"  WARNING: {pred_col} not in all_predictions.csv, skipping")
            continue

        original_acc = (pred_df[true_col] == pred_df[pred_col]).mean() * 100
        new_mean = srow['mean_accuracy']
        new_std = srow['std_accuracy']

        z_score = (original_acc - new_mean) / new_std if new_std > 0 else 0.0

        if new_std < 5:
            stability = 'STABLE'
        elif new_std <= 10:
            stability = 'MODERATE'
        else:
            stability = 'UNSTABLE'

        # --- Level 3: Error pattern overlap ---
        # Per-fold predictions not saved in npz; extract original errors
        orig_wrong_mask = pred_df[true_col] != pred_df[pred_col]
        orig_error_lds = sorted(pred_df.loc[orig_wrong_mask, 'L_D'].tolist())
        orig_error_str = '; '.join(f'{ld:.3f}' for ld in orig_error_lds)

        # Cannot compute new error patterns (per-fold preds not saved)
        new_error_str = 'N/A (per-fold predictions not saved)'
        jaccard = float('nan')
        error_overlap = 'NOT AVAILABLE'

        # --- Verdict ---
        if abs(z_score) > 2 or stability == 'UNSTABLE':
            verdict = 'SCENARIO C \u2014 UNRELIABLE'
            interpretation = ('Original result appears to be an outlier or '
                              'training is unstable. Investigate before reporting.')
        elif abs(z_score) <= 2 and stability in ['STABLE', 'MODERATE']:
            verdict = 'SCENARIO A \u2014 RELIABLE'
            interpretation = ('Original result is real and reproducible. '
                              'Safe to report as-is.')
        else:
            verdict = 'SCENARIO B \u2014 STOCHASTIC'
            interpretation = ('Aggregate accuracy reproduces but report '
                              'mean \u00b1 std instead of single number.')

        rows.append({
            'model': model_key,
            'n_classes': nc,
            'original_acc': round(original_acc, 1),
            'new_mean_acc': round(new_mean, 1),
            'new_std_acc': round(new_std, 1),
            'z_score': round(z_score, 2),
            'stability_category': stability,
            'original_error_LDs': orig_error_str,
            'new_error_LDs': new_error_str,
            'jaccard_similarity': jaccard,
            'error_overlap_category': error_overlap,
            'verdict': verdict,
            'interpretation': interpretation,
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'comparison_summary.csv'), index=False)

# --- Verdict report ---
counts = df['verdict'].value_counts()
n_a = counts.get('SCENARIO A \u2014 RELIABLE', 0)
n_b = counts.get('SCENARIO B \u2014 STOCHASTIC', 0)
n_c = counts.get('SCENARIO C \u2014 UNRELIABLE', 0)

lines = []
lines.append('# Deep Learning Reliability Check \u2014 Original vs New Runs\n\n')

lines.append('## Headline\n\n')
lines.append(f'Of 8 (model \u00d7 class scheme) combinations tested:\n')
lines.append(f'- **{n_a}** were SCENARIO A (reliable)\n')
lines.append(f'- **{n_b}** were SCENARIO B (stochastic)\n')
lines.append(f'- **{n_c}** were SCENARIO C (unreliable)\n\n')

lines.append('> Note: Per-fold predictions were not saved in step 2, so '
             'Level 3 error-pattern overlap (Jaccard) could not be computed. '
             'Verdicts are based on accuracy consistency and seed stability only.\n\n')

lines.append('## Per-Model Breakdown\n\n')
for model_key in ['1DCNN', 'LSTM', 'GRU', '2DCNN_RP']:
    lines.append(f'### {model_key}\n')
    sub = df[df['model'] == model_key]
    for _, r in sub.iterrows():
        nc = int(r['n_classes'])
        lines.append(
            f'- **{nc}-class**: ORIGINAL {r["original_acc"]:.0f}%, '
            f'NEW mean {r["new_mean_acc"]:.0f}% \u00b1 {r["new_std_acc"]:.1f}% '
            f'(5 seeds). Z={r["z_score"]:+.2f}. {r["stability_category"]}. '
            f'**VERDICT: {r["verdict"]}**\n')
        if r['original_error_LDs']:
            lines.append(f'  - Original errors at L/D: {r["original_error_LDs"]}\n')
    lines.append('\n')

# What to report
lines.append('## What to Report in the PPT\n\n')

scenario_a = df[df['verdict'].str.contains('RELIABLE')]
scenario_b = df[df['verdict'].str.contains('STOCHASTIC')]
scenario_c = df[df['verdict'].str.contains('UNRELIABLE')]

lines.append('**Report with confidence (Scenario A):**\n')
if len(scenario_a) > 0:
    for _, r in scenario_a.iterrows():
        lines.append(f'- {r["model"]} {int(r["n_classes"])}-class: '
                      f'{r["original_acc"]:.0f}%\n')
else:
    lines.append('- (none)\n')

lines.append('\n**Report with caveats (Scenario B):**\n')
if len(scenario_b) > 0:
    for _, r in scenario_b.iterrows():
        lines.append(f'- {r["model"]} {int(r["n_classes"])}-class: '
                      f'report as {r["new_mean_acc"]:.0f}% \u00b1 '
                      f'{r["new_std_acc"]:.1f}%\n')
else:
    lines.append('- (none)\n')

lines.append('\n**Do not report without investigation (Scenario C):**\n')
if len(scenario_c) > 0:
    for _, r in scenario_c.iterrows():
        lines.append(f'- {r["model"]} {int(r["n_classes"])}-class: '
                      f'original {r["original_acc"]:.0f}% vs new '
                      f'{r["new_mean_acc"]:.0f}% \u00b1 {r["new_std_acc"]:.1f}% '
                      f'(z={r["z_score"]:+.2f})\n')
else:
    lines.append('- (none)\n')

# Honest framing
lines.append('\n## Honest Framing Suggestions\n\n')

for _, r in df.iterrows():
    model = r['model']
    nc = int(r['n_classes'])
    if 'RELIABLE' in r['verdict']:
        lines.append(
            f'**{model} ({nc}-class):** "{model} achieved {r["original_acc"]:.0f}% '
            f'accuracy on the {nc}-class task. Retraining with 5 random seeds '
            f'produced {r["new_mean_acc"]:.0f}% \u00b1 {r["new_std_acc"]:.1f}%, '
            f'confirming this result is reproducible."\n\n')
    elif 'STOCHASTIC' in r['verdict']:
        lines.append(
            f'**{model} ({nc}-class):** "Across 5 random seeds, {model} achieved '
            f'{r["new_mean_acc"]:.0f}% \u00b1 {r["new_std_acc"]:.1f}% on the '
            f'{nc}-class task, indicating moderate sensitivity to initialization."\n\n')
    else:
        lines.append(
            f'**{model} ({nc}-class):** "The originally reported {r["original_acc"]:.0f}% '
            f'accuracy was not consistently reproduced across seeds '
            f'({r["new_mean_acc"]:.0f}% \u00b1 {r["new_std_acc"]:.1f}%). '
            f'This result should be interpreted with caution given the high '
            f'variance from random initialization."\n\n')

with open(os.path.join(OUT, 'verdict_report.md'), 'w') as f:
    f.writelines(lines)

# --- Console output ---
print('\n' + '=' * 75)
print('  DEEP LEARNING RELIABILITY CHECK')
print('=' * 75)
print(f'\n  Scenario A (reliable):   {n_a}/8')
print(f'  Scenario B (stochastic): {n_b}/8')
print(f'  Scenario C (unreliable): {n_c}/8\n')

print(f'  {"Model":<12} {"nc":>3} {"Orig":>6} {"Mean":>6} {"Std":>5} '
      f'{"Z":>6} {"Stability":<10} {"Verdict"}')
print('  ' + '-' * 72)
for _, r in df.iterrows():
    short_verdict = 'A-RELIABLE' if 'RELIABLE' in r['verdict'] else (
        'B-STOCHASTIC' if 'STOCHASTIC' in r['verdict'] else 'C-UNRELIABLE')
    print(f'  {r["model"]:<12} {int(r["n_classes"]):>3} '
          f'{r["original_acc"]:>5.0f}% {r["new_mean_acc"]:>5.0f}% '
          f'{r["new_std_acc"]:>4.1f}% {r["z_score"]:>+5.2f} '
          f'{r["stability_category"]:<10} {short_verdict}')

print(f'\n  Saved: {os.path.join(OUT, "comparison_summary.csv")}')
print(f'  Saved: {os.path.join(OUT, "verdict_report.md")}')
