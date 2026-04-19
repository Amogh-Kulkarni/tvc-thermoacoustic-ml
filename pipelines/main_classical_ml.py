"""
Classical ML Classification Pipeline for TVC Thermoacoustic Regimes
=====================================================================
This is the primary analysis script. It:
1. Loads data (real .mat files or synthetic demo data)
2. Extracts physics-informed features from each recording
3. Extracts nonlinear dynamics features (0-1 test, Poincare, autocorrelation)
4. Trains SVM, Random Forest, and XGBoost classifiers
5. Evaluates using leave-one-L/D-out cross-validation
6. Generates confusion matrices, feature importance, and summary tables

Usage:
    python main_classical_ml.py              # Uses synthetic demo data
    python main_classical_ml.py --real-data   # Uses your actual .mat files
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Add project source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import extract_recording_features
from nonlinear_features import compute_all_nonlinear_features
from data_loading import (
    load_all_data, create_demo_data, REGIME_LABELS, SAMPLING_FREQ
)

# Try importing xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using only SVM and Random Forest")

# 3-class regime labels: merge Limit Cycle + Period-2 -> Periodic,
# SNA + Chaos -> Aperiodic
REGIME_LABELS_3CLASS = {
    0: "Periodic",
    1: "Quasi-periodic",
    2: "Aperiodic"
}


def remap_to_3class(y):
    """Remap 5-class labels to 3 classes.
    0,1 (Limit Cycle, Period-2) -> 0 (Periodic)
    2 (Quasi-periodic)          -> 1
    3,4 (SNA, Chaos)            -> 2 (Aperiodic)
    """
    mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    return np.array([mapping[label] for label in y])


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def build_windowed_feature_matrix(dataset, fs=SAMPLING_FREQ, window_ms=50, overlap=0.5):
    """Extract windowed features (mean + std across windows) for each recording."""
    print("\n=== Extracting Windowed Features ===")

    all_features = []
    all_labels = []
    all_LD = []

    for i, rec in enumerate(dataset):
        print(f"  Processing {rec['filename']} (L/D={rec['LD_ratio']:.3f}, "
              f"{rec['regime_name']})...")

        recording_features, window_features_list = extract_recording_features(
            rec['pressure'], fs, window_ms=window_ms, overlap=overlap
        )

        all_features.append(recording_features)
        all_labels.append(rec['regime_label'])
        all_LD.append(rec['LD_ratio'])

        print(f"    -> {len(window_features_list)} windows, "
              f"{len(recording_features)} features")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_labels)
    LD_values = np.array(all_LD)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Windowed feature matrix: {X.shape}")
    return X, y, feature_names, LD_values


def build_nonlinear_feature_matrix(dataset, fs=SAMPLING_FREQ):
    """Extract nonlinear dynamics features from full recordings."""
    print("\n=== Extracting Nonlinear Dynamics Features ===")
    print("  (0-1 test, Poincare return map, autocorrelation from full recordings)")

    all_features = []

    for i, rec in enumerate(dataset):
        print(f"  [{i+1:2d}/{len(dataset)}] {rec['filename']} "
              f"(L/D={rec['LD_ratio']:.3f})...", end="", flush=True)

        nl_features = compute_all_nonlinear_features(rec['pressure'], fs)
        all_features.append(nl_features)

        k_ch1 = nl_features.get('ch1_z1_K', 0.0)
        print(f" K={k_ch1:.3f}")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Nonlinear feature matrix: {X.shape}")
    return X, feature_names, all_features


# =============================================================================
# LEAVE-ONE-L/D-OUT CROSS-VALIDATION
# =============================================================================

def leave_one_out_cv(X, y, LD_values, classifier_factory, classifier_name="Classifier",
                     regime_labels=None, quiet=False):
    """Perform leave-one-L/D-out cross-validation."""
    if regime_labels is None:
        regime_labels = REGIME_LABELS

    unique_LD = np.unique(LD_values)
    n_folds = len(unique_LD)

    all_preds = []
    all_true = []
    all_LD_test = []
    trained_models = []

    if not quiet:
        print(f"\n=== Leave-One-L/D-Out CV: {classifier_name} ({n_folds} folds) ===")

    for fold, held_out_LD in enumerate(unique_LD):
        train_mask = LD_values != held_out_LD
        test_mask = LD_values == held_out_LD

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = classifier_factory()
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_LD_test.extend([held_out_LD] * len(y_test))
        trained_models.append((clf, scaler))

        if not quiet:
            correct = "OK" if y_pred[0] == y_test[0] else "MISS"
            true_name = regime_labels[y_test[0]]
            pred_name = regime_labels[y_pred[0]]
            print(f"  Fold {fold+1:2d}: L/D={held_out_LD:.3f} | "
                  f"True: {true_name:15s} | Pred: {pred_name:15s} | {correct}")

    accuracy = accuracy_score(all_true, all_preds)
    if not quiet:
        print(f"\n  Overall Accuracy: {accuracy:.1%} "
              f"({sum(np.array(all_true)==np.array(all_preds))}/{len(all_true)})")

    return {
        'y_true': np.array(all_true),
        'y_pred': np.array(all_preds),
        'LD_test': np.array(all_LD_test),
        'accuracy': accuracy,
        'models': trained_models,
        'classifier_name': classifier_name
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_confusion_matrix(results, regime_labels=None, save_path=None):
    """Plot confusion matrix with regime names."""
    if regime_labels is None:
        regime_labels = REGIME_LABELS

    present_classes = sorted(set(results['y_true']) | set(results['y_pred']))
    present_names = [regime_labels[c] for c in present_classes]

    cm = confusion_matrix(results['y_true'], results['y_pred'], labels=present_classes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_xlabel('Predicted Regime')
    ax.set_ylabel('True Regime')
    ax.set_title(f"{results['classifier_name']}\n"
                 f"Leave-One-L/D-Out CV Accuracy: {results['accuracy']:.1%}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_feature_importance(X, y, feature_names, top_n=20, save_path=None):
    """Plot feature importance with color coding for feature types."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = []
    for idx in indices:
        name = feature_names[idx]
        if any(k in name for k in ['z1_K', 'poincare', 'autocorr_decay_10',
                                     'autocorr_peak_ratio', 'nl_rms', 'nl_dom_freq']):
            colors.append('#F44336')   # Red = nonlinear dynamics
        elif 'std_' in name:
            colors.append('#FF9800')   # Orange = windowed variability
        elif any(k in name for k in ['ch1_ch2', 'ch1_ch3', 'ch2_ch3']):
            colors.append('#9C27B0')   # Purple = cross-channel
        else:
            colors.append('#2196F3')   # Blue = windowed single-channel mean

    ax.barh(range(top_n), importances[indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel('Feature Importance (Random Forest)')
    ax.set_title(f'Top {top_n} Features for Regime Classification (Combined Feature Set)')
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', label='Nonlinear dynamics (0-1 test, Poincare, etc.)'),
        Patch(facecolor='#2196F3', label='Windowed single-channel (mean)'),
        Patch(facecolor='#FF9800', label='Windowed variability (std)'),
        Patch(facecolor='#9C27B0', label='Windowed cross-channel'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig, feature_names, importances


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(use_real_data=False, data_dir="./data", results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)

    # ===================== STEP 1: LOAD DATA =====================
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    if use_real_data:
        dataset = load_all_data(data_dir=data_dir)
    else:
        print("Using synthetic demo data (run with --real-data for actual .mat files)")
        dataset = create_demo_data()

    # ===================== STEP 2: WINDOWED FEATURES =====================
    print("\n" + "=" * 70)
    print("STEP 2: WINDOWED FEATURE EXTRACTION (66 features)")
    print("=" * 70)

    X_win, y, feature_names_win, LD_values = build_windowed_feature_matrix(dataset)

    # ===================== STEP 3: NONLINEAR FEATURES =====================
    print("\n" + "=" * 70)
    print("STEP 3: NONLINEAR DYNAMICS FEATURE EXTRACTION")
    print("=" * 70)

    X_nl, feature_names_nl, nl_features_list = build_nonlinear_feature_matrix(dataset)

    # Print K-value verification table
    print("\n  --- 0-1 Test K-value Verification ---")
    print(f"  {'L/D':>6s}  {'K (ch1)':>8s}  {'Regime':>15s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*15}")
    for i, rec in enumerate(dataset):
        k_val = nl_features_list[i].get('ch1_z1_K', 0.0)
        print(f"  {rec['LD_ratio']:6.3f}  {k_val:8.3f}  {rec['regime_name']:>15s}")

    # ===================== BUILD COMBINED FEATURES =====================
    X_combined = np.hstack([X_win, X_nl])
    feature_names_combined = list(feature_names_win) + list(feature_names_nl)
    print(f"\n  Combined feature matrix: {X_combined.shape} "
          f"({len(feature_names_win)} windowed + {len(feature_names_nl)} nonlinear)")

    # 3-class labels
    y_3class = remap_to_3class(y)

    # ===================== DEFINE CLASSIFIERS =====================
    classifiers = {
        'SVM (RBF)': lambda: SVC(kernel='rbf', C=10, gamma='scale',
                                  probability=True, random_state=42),
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=2,
            random_state=42),
    }
    if HAS_XGBOOST:
        classifiers['XGBoost'] = lambda: XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_child_weight=2, random_state=42,
            eval_metric='mlogloss', verbosity=0)

    clf_names = list(classifiers.keys())

    # ===================== RUN ALL EXPERIMENTS =====================
    experiments = [
        ("Windowed only (66)",  "5class", X_win,      feature_names_win,      y,        REGIME_LABELS,        "5c_win"),
        ("Nonlinear only",      "5class", X_nl,       feature_names_nl,       y,        REGIME_LABELS,        "5c_nl"),
        ("Combined",            "5class", X_combined,  feature_names_combined, y,        REGIME_LABELS,        "5c_comb"),
        ("Windowed only (66)",  "3class", X_win,      feature_names_win,      y_3class, REGIME_LABELS_3CLASS, "3c_win"),
        ("Nonlinear only",      "3class", X_nl,       feature_names_nl,       y_3class, REGIME_LABELS_3CLASS, "3c_nl"),
        ("Combined",            "3class", X_combined,  feature_names_combined, y_3class, REGIME_LABELS_3CLASS, "3c_comb"),
    ]

    # Collect accuracies for summary table
    summary_rows = []

    for exp_name, n_classes, X_exp, feat_names, y_exp, rlabels, tag in experiments:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {exp_name} | {n_classes} | {X_exp.shape[1]} features")
        print("=" * 70)

        # Show per-fold detail only for 5-class combined
        verbose = (tag == "5c_comb")

        row = {"Feature Set": exp_name, "Classes": n_classes}

        for clf_name, clf_factory in classifiers.items():
            results = leave_one_out_cv(
                X_exp, y_exp, LD_values, clf_factory,
                classifier_name=f"{clf_name} [{exp_name}, {n_classes}]",
                regime_labels=rlabels,
                quiet=not verbose
            )

            acc = results['accuracy']
            row[clf_name] = acc

            if not verbose:
                print(f"  {clf_name}: {acc:.1%}")

            # Save confusion matrix
            safe_tag = tag + "_" + clf_name.replace(' ', '_').replace('(', '').replace(')', '')
            plot_confusion_matrix(
                results, regime_labels=rlabels,
                save_path=os.path.join(results_dir, f'confusion_{safe_tag}.png')
            )

            # Print classification report for combined 5-class
            if verbose:
                present_classes = sorted(set(results['y_true']))
                target_names = [rlabels[c] for c in present_classes]
                print(classification_report(
                    results['y_true'], results['y_pred'],
                    labels=present_classes, target_names=target_names,
                    zero_division=0
                ))

        summary_rows.append(row)

    # ===================== FEATURE IMPORTANCE (combined 5-class) =====================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Combined Feature Set, 5-class)")
    print("=" * 70)

    fig, fnames, importances = plot_feature_importance(
        X_combined, y, feature_names_combined, top_n=20,
        save_path=os.path.join(results_dir, 'feature_importance_combined.png')
    )

    # Print top 20 features with importance values
    indices = np.argsort(importances)[::-1][:20]
    print(f"\n  {'Rank':>4s}  {'Feature':>40s}  {'Importance':>10s}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*10}")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:4d}  {fnames[idx]:>40s}  {importances[idx]:10.4f}")

    # ===================== SUMMARY COMPARISON TABLE =====================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 70)

    # Header
    header = f"  {'Feature Set':<25s} {'Classes':>7s}"
    for cn in clf_names:
        header += f"  {cn:>14s}"
    print(header)
    print("  " + "-" * (25 + 7 + 14 * len(clf_names) + 2 * len(clf_names)))

    for row in summary_rows:
        line = f"  {row['Feature Set']:<25s} {row['Classes']:>7s}"
        for cn in clf_names:
            line += f"  {row[cn]:>13.1%}"
        print(line)

    # Save features
    np.savez(os.path.join(results_dir, 'features.npz'),
             X_win=X_win, X_nl=X_nl, X_combined=X_combined,
             y=y, y_3class=y_3class,
             feature_names_win=feature_names_win,
             feature_names_nl=feature_names_nl,
             feature_names_combined=feature_names_combined,
             LD_values=LD_values)

    print(f"\n  All results saved to: {results_dir}/")
    print(f"  Files generated:")
    for f in sorted(os.listdir(results_dir)):
        print(f"    - {f}")

    return summary_rows


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    use_real = "--real-data" in sys.argv

    data_dir = "./data"
    results_dir = "./results"

    for arg in sys.argv[1:]:
        if arg.startswith("--data-dir="):
            data_dir = arg.split("=")[1]
        elif arg.startswith("--results-dir="):
            results_dir = arg.split("=")[1]

    run_pipeline(
        use_real_data=use_real,
        data_dir=data_dir,
        results_dir=results_dir
    )
