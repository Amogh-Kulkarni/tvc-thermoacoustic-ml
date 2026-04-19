"""
Step 3: Run three CV experiments using the EXISTING leave_one_out_cv.

    Experiment 1  baseline only  X_combined (93 features)
    Experiment 2  NLD2 only       X_nld2     (27 features)
    Experiment 3  combined        hstack     (120 features)

Each experiment is evaluated with SVM / Random Forest / XGBoost at both
5-class and 3-class resolution.

CRITICAL: The imported leave_one_out_cv function already fits a
StandardScaler inside each fold (fit on training, transform on test).
Do NOT wrap classifier factories in an outer StandardScaler pipeline
here -- that would double-scale. Pass raw classifier factories.
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Import the existing CV function -- this is the key reuse point.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main_classical_ml import leave_one_out_cv  # noqa: E402

np.random.seed(42)

# =========================================================================
# Classifier factories -- match main_classical_ml.py EXACTLY
# =========================================================================

def make_svm():
    return SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=42)

def make_rf():
    return RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)

def make_xgb():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_child_weight=2, random_state=42,
        eval_metric='mlogloss', verbosity=0)


CLASSIFIERS = {
    'SVM_RBF':       make_svm,
    'Random_Forest': make_rf,
}
if HAS_XGBOOST:
    CLASSIFIERS['XGBoost'] = make_xgb
else:
    print("WARNING: xgboost not available, running SVM and RF only")


REGIME_LABELS_5C = {
    0: 'Limit Cycle', 1: 'Period-2', 2: 'Quasi-periodic',
    3: 'SNA', 4: 'Chaos',
}
REGIME_LABELS_3C = {0: 'Periodic', 1: 'Quasi-periodic', 2: 'Aperiodic'}


# =========================================================================
# Helpers
# =========================================================================

def plot_cm(result, labels_dict, save_path, title_suffix=""):
    pc = sorted(set(result['y_true']) | set(result['y_pred']))
    names = [labels_dict[c] for c in pc]
    cm = confusion_matrix(result['y_true'], result['y_pred'], labels=pc)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f"{result['classifier_name']}{title_suffix}\n"
                 f"Accuracy: {result['accuracy']*100:.1f}%")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_feature_importance(X, y, fnames, save_path, title, top_n=20):
    """Refit RF on full (scaled) data to extract importance for visualization."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = make_rf()
    rf.fit(Xs, y)
    imp = pd.DataFrame({'feature': fnames, 'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False).reset_index(drop=True)

    top = imp.head(top_n)
    colors = ['#FF9500' if str(f).startswith('nld2_') else '#0C5DA5'
              for f in top['feature']]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top) + 1)))
    ax.barh(range(len(top)), top['importance'], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Random Forest importance')
    ax.set_title(title)

    from matplotlib.patches import Patch
    handles = [
        Patch(color='#0C5DA5', label='baseline feature'),
        Patch(color='#FF9500', label='NLD2 feature'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    return imp


# =========================================================================
# Main
# =========================================================================

def main():
    # ---- Load feature matrices ----
    baseline_path = os.path.join('..', 'results', 'features.npz')
    nld2_path = os.path.join('results', 'nld2_aligned.npz')

    if not os.path.exists(baseline_path):
        print(f"ERROR: {baseline_path} not found. Run main_classical_ml.py first.")
        sys.exit(1)
    if not os.path.exists(nld2_path):
        print(f"ERROR: {nld2_path} not found. Run step2 first.")
        sys.exit(1)

    baseline = np.load(baseline_path, allow_pickle=True)
    X_baseline = baseline['X_combined']
    y_5c = baseline['y']
    y_3c = baseline['y_3class']
    LD = baseline['LD_values']
    feature_names_baseline = list(baseline['feature_names_combined'])

    nld2 = np.load(nld2_path, allow_pickle=True)
    X_nld2 = nld2['X_nld2']
    feature_names_nld2 = list(nld2['feature_names_nld2'])
    assert np.allclose(LD, nld2['LD_values']), "L/D ordering mismatch between baseline and NLD2"

    X_combined = np.hstack([X_baseline, X_nld2])
    feature_names_combined = feature_names_baseline + feature_names_nld2

    print("=" * 70)
    print("Feature matrix shapes")
    print("=" * 70)
    print(f"  Baseline (main pipeline 'combined'): {X_baseline.shape}")
    print(f"  NLD2 (aligned):                       {X_nld2.shape}")
    print(f"  Combined (baseline + NLD2):           {X_combined.shape}")
    print(f"  5-class labels: {np.bincount(y_5c)}")
    print(f"  3-class labels: {np.bincount(y_3c)}")

    EXPERIMENTS = {
        'experiment_1_baseline':  (X_baseline, feature_names_baseline),
        'experiment_2_nld2_only': (X_nld2,     feature_names_nld2),
        'experiment_3_combined':  (X_combined, feature_names_combined),
    }

    all_results = {}
    headline = []

    for exp_name, (X, fnames) in EXPERIMENTS.items():
        out_dir = os.path.join('results', exp_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'=' * 70}")
        print(f"  {exp_name}: X shape = {X.shape}")
        print(f"{'=' * 70}")

        exp_results = {}

        for clf_name, factory in CLASSIFIERS.items():
            # 5-class CV
            res_5c = leave_one_out_cv(
                X, y_5c, LD,
                classifier_factory=factory,
                classifier_name=f"{clf_name} (5c)",
                regime_labels=REGIME_LABELS_5C,
                quiet=True,
            )
            # 3-class CV
            res_3c = leave_one_out_cv(
                X, y_3c, LD,
                classifier_factory=factory,
                classifier_name=f"{clf_name} (3c)",
                regime_labels=REGIME_LABELS_3C,
                quiet=True,
            )

            entry = {
                '5c_accuracy': res_5c['accuracy'],
                '3c_accuracy': res_3c['accuracy'],
                '5c_y_true': res_5c['y_true'],
                '5c_y_pred': res_5c['y_pred'],
                '3c_y_true': res_3c['y_true'],
                '3c_y_pred': res_3c['y_pred'],
                'LD_test': res_5c['LD_test'],
            }
            exp_results[clf_name] = entry
            print(f"  {clf_name:15s}  5c: {res_5c['accuracy']*100:5.1f}%"
                  f"  3c: {res_3c['accuracy']*100:5.1f}%")

            headline.append({
                'experiment': exp_name, 'classifier': clf_name,
                'accuracy_5c': res_5c['accuracy'],
                'accuracy_3c': res_3c['accuracy'],
            })

            # Save confusion matrices
            plot_cm(res_5c, REGIME_LABELS_5C,
                    os.path.join(out_dir, f'confusion_5c_{clf_name}.png'))
            plot_cm(res_3c, REGIME_LABELS_3C,
                    os.path.join(out_dir, f'confusion_3c_{clf_name}.png'))

            # Save predictions CSV
            pred_df = pd.DataFrame({
                'L_D':      entry['LD_test'],
                'true_5c':  entry['5c_y_true'],
                'pred_5c':  entry['5c_y_pred'],
                'correct_5c': entry['5c_y_true'] == entry['5c_y_pred'],
                'true_3c':  entry['3c_y_true'],
                'pred_3c':  entry['3c_y_pred'],
                'correct_3c': entry['3c_y_true'] == entry['3c_y_pred'],
            })
            pred_df.to_csv(os.path.join(out_dir, f'predictions_{clf_name}.csv'),
                            index=False)

        # Per-experiment accuracy table
        summary_df = pd.DataFrame(
            {clf: {'accuracy_5c': r['5c_accuracy'],
                   'accuracy_3c': r['3c_accuracy']}
             for clf, r in exp_results.items()}
        ).T.reset_index().rename(columns={'index': 'classifier'})
        summary_df.to_csv(os.path.join(out_dir, 'accuracy_summary.csv'),
                           index=False)

        # Feature importance for this experiment (RF on full data, 5-class)
        imp = plot_feature_importance(
            X, y_5c, fnames,
            save_path=os.path.join(out_dir, 'feature_importance_5c.png'),
            title=f'{exp_name}: top 20 RF features (5-class)',
        )
        imp.to_csv(os.path.join(out_dir, 'feature_importance_5c.csv'), index=False)

        all_results[exp_name] = exp_results

    # ---- Save aggregated results ----
    with open(os.path.join('results', 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    pd.DataFrame(headline).to_csv(
        os.path.join('results', 'headline_accuracies.csv'), index=False)

    # ---- Headline comparison ----
    print(f"\n{'=' * 70}")
    print("  HEADLINE COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<25} {'Classifier':<15} {'5c':>8} {'3c':>8}")
    print("-" * 60)
    for row in headline:
        print(f"{row['experiment']:<25} {row['classifier']:<15} "
              f"{row['accuracy_5c']*100:>7.1f}% {row['accuracy_3c']*100:>7.1f}%")

    print(f"\nSaved all results to results/")


if __name__ == "__main__":
    main()
