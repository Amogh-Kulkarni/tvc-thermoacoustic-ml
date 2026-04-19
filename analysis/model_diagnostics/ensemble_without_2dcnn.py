"""Check whether 3-class 100% Hard-Vote depends on the unstable 2D-CNN."""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, '3classensemblevote')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(PROJ, '..', 'results', 'all_predictions.csv'))
true = df['true_3c'].values
LD = df['L_D'].values

ENSEMBLES = {
    'All 7 models': [
        'pred_3c_SVM_comb', 'pred_3c_RF_comb', 'pred_3c_XGB_comb',
        'pred_3c_1D-CNN', 'pred_3c_LSTM', 'pred_3c_GRU', 'pred_3c_2D-CNN_RP',
    ],
    '6 models (no 2D-CNN)': [
        'pred_3c_SVM_comb', 'pred_3c_RF_comb', 'pred_3c_XGB_comb',
        'pred_3c_1D-CNN', 'pred_3c_LSTM', 'pred_3c_GRU',
    ],
    '3 classical only': [
        'pred_3c_SVM_comb', 'pred_3c_RF_comb', 'pred_3c_XGB_comb',
    ],
}

CLASS_NAMES = {0: 'Periodic', 1: 'Quasi-periodic', 2: 'Aperiodic'}
rows = []

for ens_name, cols in ENSEMBLES.items():
    votes = df[cols].values  # (20, n_models)
    preds = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        counts = np.bincount(votes[i], minlength=3)
        preds[i] = counts.argmax()  # lowest index wins ties

    correct = (preds == true)
    acc = correct.mean() * 100
    n_correct = correct.sum()
    wrong_lds = sorted(LD[~correct].tolist())
    wrong_str = '; '.join(f'{ld:.3f}' for ld in wrong_lds) if wrong_lds else '(none)'

    rows.append({
        'ensemble_name': ens_name,
        'accuracy': round(acc, 1),
        'n_correct': int(n_correct),
        'wrong_LDs': wrong_str,
    })

csv_df = pd.DataFrame(rows)
csv_df.to_csv(os.path.join(OUT, 'ensemble_accuracies.csv'), index=False)

# Build markdown
all7 = rows[0]
no2d = rows[1]
classical = rows[2]

lines = []
lines.append('# Ensemble Robustness Check \u2014 Does 3-class 100% Depend on 2D-CNN?\n\n')
lines.append('## Results\n\n')
lines.append('| Ensemble | Accuracy | Correct | Wrong L/D values |\n')
lines.append('|---|---|---|---|\n')
for r in rows:
    lines.append(f'| {r["ensemble_name"]} | {r["accuracy"]:.0f}% '
                 f'| {r["n_correct"]}/20 | {r["wrong_LDs"]} |\n')

lines.append('\n## Per-model 3-class predictions on the 20 samples\n\n')
lines.append('| L/D | True | SVM | RF | XGB | 1D-CNN | LSTM | GRU | 2D-CNN |\n')
lines.append('|---|---|---|---|---|---|---|---|---|\n')
model_cols = ['pred_3c_SVM_comb', 'pred_3c_RF_comb', 'pred_3c_XGB_comb',
              'pred_3c_1D-CNN', 'pred_3c_LSTM', 'pred_3c_GRU', 'pred_3c_2D-CNN_RP']
for i in range(len(df)):
    vals = [f'{CLASS_NAMES[int(df[c].iloc[i])]}' for c in model_cols]
    true_name = CLASS_NAMES[int(true[i])]
    marks = []
    for c in model_cols:
        v = int(df[c].iloc[i])
        name = CLASS_NAMES[v]
        marks.append(f'**{name}**' if v != true[i] else name)
    lines.append(f'| {LD[i]:.3f} | {true_name} | {" | ".join(marks)} |\n')

lines.append('\n## Interpretation\n\n')

if all7['accuracy'] == 100 and no2d['accuracy'] == 100:
    lines.append('The 100% 3-class result **does NOT depend on the 2D-CNN**. '
                 'Removing the unstable 2D-CNN still yields perfect classification. '
                 'This strengthens the reported result.\n\n')
    if classical['accuracy'] == 100:
        lines.append('Even classical ML alone reaches 100% on 3-class. '
                     'The ensemble and deep learning are not strictly necessary '
                     'for this result. Most robust possible framing.\n')
    else:
        lines.append(f'Classical models alone drop to {classical["accuracy"]:.0f}% '
                     f'(wrong at L/D: {classical["wrong_LDs"]}), so the DL models '
                     f'do contribute to the perfect ensemble.\n')
elif all7['accuracy'] == 100 and no2d['accuracy'] < 100:
    lines.append(f'The 2D-CNN contributes to the ensemble\'s perfect accuracy. '
                 f'Without it, the ensemble drops to {no2d["accuracy"]:.0f}% '
                 f'(wrong at L/D: {no2d["wrong_LDs"]}). This means the headline '
                 f'100% result relies partly on a model flagged as seed-sensitive. '
                 f'Worth noting in the PPT as a caveat.\n')
else:
    lines.append(f'The original All-7 ensemble scored {all7["accuracy"]:.0f}% '
                 f'(not 100%). Without 2D-CNN: {no2d["accuracy"]:.0f}%. '
                 f'Classical only: {classical["accuracy"]:.0f}%.\n')

with open(os.path.join(OUT, 'robustness_check.md'), 'w') as f:
    f.writelines(lines)

# Print to console
print(''.join(lines))
