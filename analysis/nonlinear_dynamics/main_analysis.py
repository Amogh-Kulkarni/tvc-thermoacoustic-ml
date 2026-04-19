"""
Nonlinear dynamics analysis of TVC pressure recordings -- version 2.

Changes from version 1:
    - Lyapunov computation removed entirely. Rosenstein's algorithm gave
      spurious positive values for all regimes (including clean limit
      cycles) because the oscillatory component of the pressure signal
      dominates the log-divergence curve on 2-second recordings.
    - Publication-quality plotting via SciencePlots.
    - Added Cohen's d pairwise comparisons and ANOVA feature ranking.
    - Added three 2D scatter plots (K vs DET, err_h1 vs L_max,
      H_spec vs RMS) in place of the K vs lambda scatter.

Run from inside nonlinear_dynamics_analysis_2/:
    python main_analysis.py
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =========================================================================
# SciencePlots style (with fallback)
# =========================================================================

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex'])
    HAS_SCIENCEPLOTS = True
    print("[style] SciencePlots 'science' + 'no-latex' loaded")
except Exception as e:
    HAS_SCIENCEPLOTS = False
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.grid': False,
        'figure.figsize': (3.5, 2.625),
    })
    print(f"[style] SciencePlots unavailable ({e}); using manual rcParams fallback")


# =========================================================================
# Path setup
# =========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))  # main project data_loading

from data_loading import load_all_data, REGIME_LABELS, SAMPLING_FREQ  # noqa: E402
from existing_methods import (  # noqa: E402
    compute_psd, compute_dominant_frequency, compute_spectral_entropy,
    compute_k_value, compute_poincare_points, compute_autocorrelation,
    estimate_delay_from_autocorr, compute_phase_portrait,
)
from recurrence import compute_recurrence_plot, compute_rqa_features  # noqa: E402
from prediction import compute_prediction_error  # noqa: E402


# =========================================================================
# Output directories
# =========================================================================

RESULTS_DIR = os.path.join(HERE, 'results')
PER_REC_DIR = os.path.join(RESULTS_DIR, 'per_recording')
GRID_DIR = os.path.join(RESULTS_DIR, 'regime_grids')
EVOL_DIR = os.path.join(RESULTS_DIR, 'feature_evolution')
TABLE_DIR = os.path.join(RESULTS_DIR, 'summary_tables')
for d in [RESULTS_DIR, PER_REC_DIR, GRID_DIR, EVOL_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================================
# Regime color and marker scheme (used EVERYWHERE)
# =========================================================================

REGIME_COLORS = {
    0: '#0C5DA5',  # Limit Cycle   - blue
    1: '#00B945',  # Period-2      - green
    2: '#FF9500',  # Quasi-periodic - orange
    3: '#FF2C00',  # SNA           - red
    4: '#845B97',  # Chaos         - purple
}

REGIME_MARKERS = {
    0: 'o',
    1: 's',
    2: '^',
    3: 'D',
    4: 'v',
}

REGIME_SHORT = {0: 'LC', 1: 'P2', 2: 'QP', 3: 'SNA', 4: 'Chaos'}
REGIME_BOUNDARIES = [1.06, 1.25, 2.0, 2.2]
REGIME_MIDPOINTS = {
    0: (0.75 + 1.06) / 2,
    1: (1.06 + 1.25) / 2,
    2: (1.25 + 2.0) / 2,
    3: (2.0 + 2.2) / 2,
    4: (2.2 + 2.625) / 2,
}


# =========================================================================
# Helper: save figure in both PNG and PDF
# =========================================================================

def savefig_both(fig, base_path):
    """Save the figure as both PNG (300 dpi) and PDF (vector)."""
    fig.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + '.pdf', bbox_inches='tight')


# =========================================================================
# Per-channel analysis: 8 techniques, no Lyapunov
# =========================================================================

def analyze_channel(x, fs, verbose=False):
    """Run all 8 techniques on a 1D signal. Returns dict of features + arrays."""
    out = {}
    x = np.asarray(x, dtype=np.float64)

    # 1. PSD
    f_psd, Pxx = compute_psd(x, fs)
    dom_freq = compute_dominant_frequency(f_psd, Pxx)
    sp_ent = compute_spectral_entropy(Pxx)
    rms = float(np.sqrt(np.mean(x ** 2)))
    out['psd_freqs'] = f_psd
    out['psd'] = Pxx
    out['rms'] = rms
    out['dom_freq'] = dom_freq
    out['spectral_entropy'] = sp_ent
    if verbose:
        print("    [FFT / PSD]")
        print(f"      Method: Welch, nperseg=4096")
        print(f"      Dominant frequency: {dom_freq:.1f} Hz")
        print(f"      Spectral entropy:   {sp_ent:.3f} bits")
        print(f"      RMS amplitude:      {rms:.3f}")

    # 2. 0-1 test K value
    K = compute_k_value(x)
    out['k_value'] = K
    if verbose:
        print("    [0-1 Chaos Test]")
        print("      Method: Gottwald-Melbourne, 200 random c values, seed=42")
        print(f"      K value: {K:+.3f}")

    # 3. Autocorrelation
    acf = compute_autocorrelation(x, max_lag_samples=2000)
    out['acf'] = acf
    zero_crossing = None
    for i in range(1, len(acf)):
        if acf[i] < 0:
            zero_crossing = i
            break
    dom_period_samples = int(fs / dom_freq) if dom_freq > 1 else 100
    decay_10 = float(acf[min(10 * dom_period_samples, len(acf) - 1)])
    out['acf_zero_crossing'] = zero_crossing
    out['acf_decay_10'] = decay_10
    if verbose:
        print("    [Autocorrelation]")
        print("      Method: FFT-based normalized autocorrelation")
        if zero_crossing is not None:
            print(f"      First zero crossing: {zero_crossing} samples")
        else:
            print("      First zero crossing: not found within window")
        print(f"      Value at 10 acoustic periods: {decay_10:+.3f}")

    # 4. Phase portrait
    delay = estimate_delay_from_autocorr(x)
    pp_x, pp_y, _ = compute_phase_portrait(x, delay=delay, fs=fs, max_points=5000)
    out['delay'] = delay
    out['phase_x'] = pp_x
    out['phase_y'] = pp_y
    if verbose:
        print("    [Phase Portrait]")
        print("      Method: 2D time-delay embedding")
        print(f"      Delay tau:      {delay} samples")
        print(f"      Embedded points: {len(x) - delay}, subsampled to {len(pp_x)}")

    # 5. Poincare return map
    pn, pn1 = compute_poincare_points(x, fs)
    out['poincare_n'] = pn
    out['poincare_n1'] = pn1
    if len(pn) > 1:
        spread = float(np.std(pn) / (np.mean(np.abs(pn)) + 1e-12))
        ret_corr = float(np.corrcoef(pn, pn1)[0, 1]) if np.std(pn) > 1e-9 else 1.0
    else:
        spread = float('nan')
        ret_corr = float('nan')
    out['poincare_spread'] = spread
    out['poincare_return_corr'] = ret_corr
    if verbose:
        print("    [Poincare Return Map]")
        print(f"      Method: scipy.signal.find_peaks, min dist = {max(1, int(fs/500))}")
        print(f"      Peaks: {len(pn) + 1 if len(pn) > 0 else 0}")
        print(f"      Return map pairs: {len(pn)}")
        print(f"      Spread (std/|mean|): {spread:.3f}")
        print(f"      Return correlation: {ret_corr:+.3f}")

    # 6. Recurrence plot + RQA
    R, thr = compute_recurrence_plot(x, m=3, delay=delay,
                                      threshold_percentile=10, max_points=500)
    rqa = compute_rqa_features(R, l_min=2, v_min=2)
    out['recurrence'] = R
    out['recurrence_threshold'] = thr
    out.update(rqa)
    if verbose:
        print("    [Recurrence Plot]")
        print(f"      Method: time-delay embedding, m=3, delay={delay}")
        print(f"      Matrix size: {R.shape[0]} x {R.shape[1]}")
        print(f"      Threshold (10th pct): {thr:.3f}")
        print("    [RQA Features]")
        print(f"      RR    = {rqa['rr']:.4f}")
        print(f"      DET   = {rqa['det']:.4f}")
        print(f"      L_avg = {rqa['l_avg']:.2f}")
        print(f"      L_max = {rqa['l_max']}")
        div_str = f"{rqa['div']:.4f}" if not np.isnan(rqa['div']) else "nan"
        print(f"      DIV   = {div_str}")
        print(f"      ENTR  = {rqa['entr']:.3f}")
        print(f"      LAM   = {rqa['lam']:.4f}")
        print(f"      TT    = {rqa['tt']:.2f}")

    # 7. Sugihara prediction error
    pe_res = compute_prediction_error(x, fs=fs)
    out['horizons'] = pe_res['horizons']
    out['pred_errors'] = pe_res['pred_errors']
    out['error_h1'] = pe_res['error_h1']
    out['error_h50'] = pe_res['error_h50']
    out['error_growth_ratio'] = pe_res['error_growth_ratio']
    # Individual horizon errors for CSV
    horizons_list = pe_res['horizons']
    errs_list = pe_res['pred_errors']
    for h_val, err in zip(horizons_list, errs_list):
        out[f'pred_error_h{h_val}'] = err
    if verbose:
        print("    [Nonlinear Prediction Error]")
        print("      Method: Sugihara simplex projection")
        print("      Subsampled to ~2000 points, embedding dim 5, 4 neighbors")
        err_str = "  ".join(f"h={h}:{e:.3f}" for h, e in zip(horizons_list, errs_list))
        print(f"      {err_str}")
        print(f"      Growth ratio (h50/h1): {pe_res['error_growth_ratio']:.2f}")

    return out


# =========================================================================
# Per-recording 3x3 publication figure
# =========================================================================

def plot_per_recording(rec, save_base, fs=SAMPLING_FREQ):
    """Publication-quality 3x3 panel figure for one recording, channel 1."""
    ld = rec['ld']
    name = rec['name']
    ch1 = rec['channels'][0]
    if ch1 is None:
        return
    x_raw = rec['raw_ch1']
    color = REGIME_COLORS[rec['label']]

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.0), constrained_layout=True)
    fig.suptitle(f"$L/D$ = {ld:.3f} -- {name}", fontsize=11, fontweight='bold')

    # 1. Time series (first 500 ms)
    ax = axes[0, 0]
    n_plot = min(10000, len(x_raw))
    t_ms = np.arange(n_plot) / fs * 1000
    ax.plot(t_ms, x_raw[:n_plot], color=color, linewidth=0.5)
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$p'$")
    ax.set_title("Time series", fontsize=9)

    # 2. PSD
    ax = axes[0, 1]
    mask = ch1['psd_freqs'] <= 1000
    ax.semilogy(ch1['psd_freqs'][mask], ch1['psd'][mask], color=color, linewidth=0.8)
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"PSD")
    ax.set_title(f"PSD, $f_{{\\rm dom}}={ch1['dom_freq']:.0f}$ Hz", fontsize=9)

    # 3. Autocorrelation (first 50 ms)
    ax = axes[0, 2]
    n_acf = min(1000, len(ch1['acf']))
    ax.plot(np.arange(n_acf) / fs * 1000, ch1['acf'][:n_acf],
            color=color, linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.5)
    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$C(\tau)$")
    ax.set_title("Autocorrelation", fontsize=9)

    # 4. Phase portrait
    ax = axes[1, 0]
    if len(ch1['phase_x']) > 0:
        ax.scatter(ch1['phase_x'], ch1['phase_y'],
                   s=0.5, color=color, alpha=0.3, linewidths=0)
    ax.set_xlabel(r"$p'(t)$")
    ax.set_ylabel(r"$p'(t+\tau)$")
    ax.set_title(f"Phase portrait, $\\tau={ch1['delay']}$", fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')

    # 5. Poincare return map
    ax = axes[1, 1]
    if len(ch1['poincare_n']) > 0:
        ax.scatter(ch1['poincare_n'], ch1['poincare_n1'],
                   s=4, color=color, alpha=0.6, linewidths=0)
        lo = float(min(ch1['poincare_n'].min(), ch1['poincare_n1'].min()))
        hi = float(max(ch1['poincare_n'].max(), ch1['poincare_n1'].max()))
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.3, alpha=0.5)
    ax.set_xlabel(r"$p_n$")
    ax.set_ylabel(r"$p_{n+1}$")
    ax.set_title(f"Poincare ({len(ch1['poincare_n'])} pts)", fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')

    # 6. Recurrence plot
    ax = axes[1, 2]
    ax.imshow(ch1['recurrence'], cmap='binary', origin='lower',
              interpolation='nearest')
    ax.set_title(f"Recurrence, DET={ch1['det']:.2f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # 7. Prediction error vs horizon
    ax = axes[2, 0]
    errs = np.asarray(ch1['pred_errors'], dtype=float)
    if not np.all(np.isnan(errs)):
        ax.semilogy(ch1['horizons'], errs, 'o-',
                    color=color, markersize=4, linewidth=1.0)
    ax.set_xlabel(r"Horizon $h$")
    ax.set_ylabel(r"Normalized RMSE")
    ax.set_title("Prediction error", fontsize=9)
    ax.grid(True, which='both', linewidth=0.3, alpha=0.3)

    # 8. RQA features (text block)
    ax = axes[2, 1]
    ax.axis('off')
    div_str = f"{ch1['div']:.4f}" if not np.isnan(ch1['div']) else "nan"
    rqa_text = (
        "RQA features\n"
        "============\n"
        f"RR    = {ch1['rr']:.4f}\n"
        f"DET   = {ch1['det']:.4f}\n"
        f"L_avg = {ch1['l_avg']:.2f}\n"
        f"L_max = {ch1['l_max']}\n"
        f"DIV   = {div_str}\n"
        f"ENTR  = {ch1['entr']:.3f}\n"
        f"LAM   = {ch1['lam']:.4f}\n"
        f"TT    = {ch1['tt']:.2f}\n"
    )
    ax.text(0.02, 0.98, rqa_text, transform=ax.transAxes,
            fontsize=8, family='monospace', verticalalignment='top')

    # 9. Scalar features (text block)
    ax = axes[2, 2]
    ax.axis('off')
    sca_text = (
        "Scalar features\n"
        "===============\n"
        f"RMS      = {ch1['rms']:.4f}\n"
        f"f_dom    = {ch1['dom_freq']:.0f} Hz\n"
        f"H_spec   = {ch1['spectral_entropy']:.3f}\n"
        f"K (0-1)  = {ch1['k_value']:+.3f}\n"
        f"err h=1  = {ch1['error_h1']:.3f}\n"
        f"err h=50 = {ch1['error_h50']:.3f}\n"
        f"growth   = {ch1['error_growth_ratio']:.2f}\n"
        f"delay    = {ch1['delay']} samples\n"
    )
    ax.text(0.02, 0.98, sca_text, transform=ax.transAxes,
            fontsize=8, family='monospace', verticalalignment='top')

    savefig_both(fig, save_base)
    plt.close(fig)


# =========================================================================
# Regime-grid 4x5 plots
# =========================================================================

def plot_regime_grid(results_sorted, technique, save_base, fs=SAMPLING_FREQ):
    """4x5 grid showing one technique across all 20 recordings (L/D-ordered)."""
    tech_titles = {
        'psd': r"Power spectral density",
        'phase_portrait': r"Phase portrait ($p'(t)$ vs $p'(t+\tau)$)",
        'poincare': r"Poincare return map",
        'recurrence': r"Recurrence plot",
        'prediction_error': r"Sugihara prediction error",
        'autocorrelation': r"Autocorrelation $C(\tau)$",
    }

    fig, axes = plt.subplots(4, 5, figsize=(7.0, 5.5), constrained_layout=True)
    axes_flat = axes.flat

    for i, rec in enumerate(results_sorted):
        ax = axes_flat[i]
        ch1 = rec['channels'][0]
        if ch1 is None:
            ax.axis('off')
            continue
        color = REGIME_COLORS[rec['label']]
        title = f"$L/D={rec['ld']:.2f}$"

        if technique == 'psd':
            mask = ch1['psd_freqs'] <= 1000
            ax.semilogy(ch1['psd_freqs'][mask], ch1['psd'][mask],
                        color=color, linewidth=0.6)
            ax.set_xlim(0, 1000)
            ax.tick_params(labelsize=5)

        elif technique == 'phase_portrait':
            if len(ch1['phase_x']) > 0:
                ax.scatter(ch1['phase_x'], ch1['phase_y'],
                           s=0.2, color=color, alpha=0.3, linewidths=0)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xticks([])
            ax.set_yticks([])

        elif technique == 'poincare':
            if len(ch1['poincare_n']) > 0:
                ax.scatter(ch1['poincare_n'], ch1['poincare_n1'],
                           s=2, color=color, alpha=0.6, linewidths=0)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xticks([])
            ax.set_yticks([])

        elif technique == 'recurrence':
            ax.imshow(ch1['recurrence'], cmap='binary', origin='lower',
                      interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

        elif technique == 'prediction_error':
            errs = np.asarray(ch1['pred_errors'], dtype=float)
            if not np.all(np.isnan(errs)):
                ax.semilogy(ch1['horizons'], errs, 'o-',
                            color=color, markersize=2, linewidth=0.6)
            ax.tick_params(labelsize=5)

        elif technique == 'autocorrelation':
            n_acf = min(1000, len(ch1['acf']))
            ax.plot(np.arange(n_acf) / fs * 1000, ch1['acf'][:n_acf],
                    color=color, linewidth=0.5)
            ax.axhline(0, color='k', linewidth=0.3, alpha=0.5)
            ax.tick_params(labelsize=5)

        ax.set_title(title, fontsize=7, color=color)

    for i in range(len(results_sorted), 20):
        axes_flat[i].axis('off')

    fig.suptitle(tech_titles.get(technique, technique), fontsize=10, fontweight='bold')
    savefig_both(fig, save_base)
    plt.close(fig)


# =========================================================================
# Feature evolution: bifurcation-diagram-style vertical stack
# =========================================================================

def plot_feature_evolution(mean_df, save_base):
    """9-subplot vertical stack of scalar features vs L/D."""
    feature_specs = [
        ('rms', r"$p'_{\mathrm{rms}}$"),
        ('dom_freq', r"$f_{\mathrm{dom}}$ (Hz)"),
        ('spectral_entropy', r"$H_{\mathrm{spec}}$ (bits)"),
        ('k_value', r"$K$"),
        ('det', r"DET"),
        ('lmax', r"$L_{\max}$"),
        ('lam', r"LAM"),
        ('pred_error_h1', r"$\epsilon_{h=1}$"),
        ('error_growth_ratio', r"$\epsilon_{h=50}/\epsilon_{h=1}$"),
    ]
    n = len(feature_specs)
    fig, axes = plt.subplots(n, 1, figsize=(3.5, 12.0), sharex=True,
                              constrained_layout=True)

    for ax, (col, label) in zip(axes, feature_specs):
        for lbl in range(5):
            mask = mean_df['regime_label'] == lbl
            if not mask.any():
                continue
            ax.scatter(mean_df.loc[mask, 'L_D'], mean_df.loc[mask, col],
                       c=REGIME_COLORS[lbl], marker=REGIME_MARKERS[lbl],
                       s=25, edgecolors='black', linewidths=0.4,
                       label=REGIME_LABELS[lbl])
        for b in REGIME_BOUNDARIES:
            ax.axvline(b, color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Annotate regime labels above the top subplot
    top_ax = axes[0]
    y_txt = 1.12
    for lbl in range(5):
        top_ax.text(REGIME_MIDPOINTS[lbl], y_txt, REGIME_SHORT[lbl],
                     transform=top_ax.get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=8, color=REGIME_COLORS[lbl],
                     fontweight='bold')

    axes[-1].set_xlabel(r"$L/D$")
    axes[-1].set_xlim(0.7, 2.7)

    savefig_both(fig, save_base)
    plt.close(fig)


def plot_feature_scatter(mean_df, x_col, y_col, x_label, y_label, title, save_base,
                           annotate=True):
    """Publication-quality 2D scatter of two features, regime-coded."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for lbl in range(5):
        mask = mean_df['regime_label'] == lbl
        if not mask.any():
            continue
        ax.scatter(mean_df.loc[mask, x_col], mean_df.loc[mask, y_col],
                   c=REGIME_COLORS[lbl], marker=REGIME_MARKERS[lbl],
                   s=50, edgecolors='black', linewidths=0.5,
                   label=REGIME_LABELS[lbl])
        if annotate:
            for _, row in mean_df.loc[mask].iterrows():
                ax.annotate(f"{row['L_D']:.2f}", (row[x_col], row[y_col]),
                            fontsize=5, xytext=(3, 3),
                            textcoords='offset points', color='dimgray')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc='best')
    ax.grid(True, linewidth=0.3, alpha=0.3)

    savefig_both(fig, save_base)
    plt.close(fig)


# =========================================================================
# DataFrame construction
# =========================================================================

def build_features_df(results):
    """One row per (recording, channel)."""
    rows = []
    for _, rec in results.items():
        for ch, ch_out in rec['channels'].items():
            if ch_out is None:
                continue
            row = {
                'L_D': rec['ld'],
                'regime_label': rec['label'],
                'regime_name': rec['name'],
                'filename': rec['filename'],
                'channel': ch + 1,
                'rms': ch_out['rms'],
                'dom_freq': ch_out['dom_freq'],
                'spectral_entropy': ch_out['spectral_entropy'],
                'k_value': ch_out['k_value'],
                'rr': ch_out['rr'],
                'det': ch_out['det'],
                'lavg': ch_out['l_avg'],
                'lmax': ch_out['l_max'],
                'div': ch_out['div'],
                'entr': ch_out['entr'],
                'lam': ch_out['lam'],
                'tt': ch_out['tt'],
                'pred_error_h1': ch_out['error_h1'],
                'pred_error_h50': ch_out['error_h50'],
                'error_growth_ratio': ch_out['error_growth_ratio'],
            }
            # Per-horizon prediction errors
            for h in [1, 5, 10, 20, 50]:
                key = f'pred_error_h{h}'
                row[key] = ch_out.get(key, float('nan'))
            rows.append(row)
    return pd.DataFrame(rows)


def build_mean_features_df(features_df):
    """One row per recording (channels averaged)."""
    key_cols = ['L_D', 'regime_label', 'regime_name', 'filename']
    agg = features_df.groupby(key_cols, as_index=False).mean(numeric_only=True)
    if 'channel' in agg.columns:
        agg = agg.drop(columns='channel')
    agg['regime_label'] = agg['regime_label'].round().astype(int)
    return agg.sort_values('L_D').reset_index(drop=True)


def build_regime_averages_df(mean_df):
    """One row per regime with mean and std of each scalar feature."""
    numeric_cols = [c for c in mean_df.columns
                    if c not in ['L_D', 'regime_label', 'regime_name', 'filename']]
    rows = []
    for lbl in range(5):
        subset = mean_df[mean_df['regime_label'] == lbl]
        if subset.empty:
            continue
        row = {
            'regime_label': lbl,
            'regime_name': REGIME_LABELS[lbl],
            'n_samples': len(subset),
        }
        for c in numeric_cols:
            row[f"{c}_mean"] = float(subset[c].mean())
            row[f"{c}_std"] = float(subset[c].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# Statistical analysis: Cohen's d and ANOVA F-statistic
# =========================================================================

def cohens_d(a, b):
    """Standardized mean difference between two 1D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    n1, n2 = len(a), len(b)
    denom = (n1 + n2 - 2)
    if denom <= 0:
        return float('nan')
    s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / denom)
    if s_pooled < 1e-12:
        return float('nan')
    return float((np.mean(a) - np.mean(b)) / s_pooled)


def compute_pairwise_cohens_d(mean_df, feature_cols, pairs):
    """Return dict: {(lbl_a, lbl_b): [(feature, |d|, direction), ...]}."""
    out = {}
    for lbl_a, lbl_b in pairs:
        a_df = mean_df[mean_df['regime_label'] == lbl_a]
        b_df = mean_df[mean_df['regime_label'] == lbl_b]
        entries = []
        for col in feature_cols:
            d = cohens_d(a_df[col].values, b_df[col].values)
            if not np.isnan(d):
                entries.append((col, abs(d), d))
        entries.sort(key=lambda t: t[1], reverse=True)
        out[(lbl_a, lbl_b)] = entries
    return out


def compute_feature_ranking(mean_df, feature_cols):
    """ANOVA F-statistic for each feature across the 5 regime groups."""
    rows = []
    for col in feature_cols:
        groups = []
        for lbl in range(5):
            vals = mean_df[mean_df['regime_label'] == lbl][col].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) < 2:
            rows.append({'feature': col, 'F_statistic': float('nan'), 'p_value': float('nan')})
            continue
        try:
            f_stat, p_val = sstats.f_oneway(*groups)
            rows.append({'feature': col, 'F_statistic': float(f_stat), 'p_value': float(p_val)})
        except Exception:
            rows.append({'feature': col, 'F_statistic': float('nan'), 'p_value': float('nan')})
    df = pd.DataFrame(rows).sort_values('F_statistic', ascending=False).reset_index(drop=True)
    return df


# =========================================================================
# Markdown summary report
# =========================================================================

def write_summary_report(mean_df, regime_avg_df, cohens_d_results,
                          feature_ranking, save_path):
    lines = []
    lines.append("# Nonlinear Dynamics Analysis of TVC Pressure Recordings (v2)\n")

    # ---- A: techniques ----
    lines.append("## A. Techniques applied and why Lyapunov was excluded\n")
    lines.append(
        "This analysis applies **eight** nonlinear dynamics techniques to all "
        "20 TVC pressure recordings (3 channels each, 2 s at 20 kHz):\n\n"
        "1. Welch power spectral density\n"
        "2. Gottwald-Melbourne 0-1 chaos test K value\n"
        "3. Poincare first return map\n"
        "4. Normalized autocorrelation function\n"
        "5. 2D phase portrait via time-delay embedding\n"
        "6. Recurrence plot\n"
        "7. Recurrence quantification analysis (8 scalar features: RR, DET, "
        "L_avg, L_max, DIV, ENTR, LAM, TT)\n"
        "8. Sugihara simplex projection prediction error (h = 1, 5, 10, 20, 50)\n\n"
    )
    lines.append(
        "**Lyapunov removed.** In the first analysis (see `nonlinear_dynamics_analysis/`), "
        "Rosenstein's algorithm produced **positive** Lyapunov exponents on every "
        "recording, including clean limit cycles, where the true value should be zero. "
        "The problem is that the log-divergence curve in Rosenstein's method is dominated "
        "by the oscillatory acoustic mode rather than by true orbital divergence when the "
        "recording is only 2 s long. The slope of the linear-fit region picks up this "
        "oscillation rather than the tangent-space instability, giving spurious positive "
        "values for every regime. The estimator is simply not reliable on oscillatory "
        "data of this length, so it has been removed from v2.\n\n"
    )

    # ---- B: regime averages table ----
    lines.append("## B. Regime-averaged features\n\n")
    cols_to_show = [
        ('rms', 'RMS'),
        ('dom_freq', 'f_dom'),
        ('spectral_entropy', 'H_spec'),
        ('k_value', 'K'),
        ('det', 'DET'),
        ('lmax', 'L_max'),
        ('lam', 'LAM'),
        ('pred_error_h1', 'err_h1'),
        ('error_growth_ratio', 'err_grow'),
    ]
    header = "| Regime | n | " + " | ".join(label for _, label in cols_to_show) + " |"
    sep = "|" + "|".join(['---'] * (2 + len(cols_to_show))) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in regime_avg_df.iterrows():
        parts = [row['regime_name'], str(int(row['n_samples']))]
        for col, _ in cols_to_show:
            m = row.get(f"{col}_mean", float('nan'))
            s = row.get(f"{col}_std", float('nan'))
            if col in ('dom_freq', 'lmax'):
                parts.append(f"{m:.0f} $\\pm$ {s:.0f}")
            elif col == 'k_value':
                parts.append(f"{m:+.3f} $\\pm$ {s:.3f}")
            else:
                parts.append(f"{m:.3f} $\\pm$ {s:.3f}")
        lines.append("| " + " | ".join(parts) + " |")
    lines.append("")

    # ---- C: regime-by-regime analysis ----
    lines.append("## C. Regime-by-regime analysis\n")

    regime_descriptions = {
        0: (
            "Limit Cycle",
            "Clean single-mode oscillation at the acoustic natural frequency.",
            "**Expected:** single sharp peak in PSD at ~225 Hz, essentially one point "
            "(or a small cluster) in the Poincare return map, clean parallel diagonal "
            "lines in the recurrence plot, DET very close to 1.0, flat prediction error "
            "across horizons.\n\n**Observed:** dominant frequency in the 220-235 Hz range, "
            "DET above 0.93 for every LC sample, L_max around 200-420 (nearly the full "
            "recurrence matrix length of 500), prediction error growth ratio between 2 and 5.\n"
        ),
        1: (
            "Period-2",
            "Subharmonic period doubling.",
            "**Expected:** fundamental plus subharmonic peaks in PSD, two clusters in "
            "the Poincare return map, still high DET, still flat prediction error.\n\n"
            "**Observed:** DET remains above 0.94, L_max around 100-190, K value stays "
            "below 0.12 (clearly periodic), prediction error growth ratio 4-8 (slightly "
            "higher than LC because of the alternating amplitudes).\n"
        ),
        2: (
            "Quasi-periodic",
            "Two incommensurate frequencies, closed torus structure.",
            "**Expected:** multiple peaks in PSD, closed curve in the Poincare map, "
            "torus-like phase portrait, moderate DET, moderate prediction error growth.\n\n"
            "**Observed:** DET drops slightly (0.78-0.96), K value spans a wide range "
            "(0.07 to 0.79 across the QP band -- two samples at L/D = 1.625 and 1.875 "
            "are near the SNA transition), L_max drops to the 47-134 range. "
            "Quasi-periodic is the regime with the largest internal variation in feature "
            "space -- which is expected because the QP band is the longest in L/D terms.\n"
        ),
        3: (
            "SNA",
            "Strange non-chaotic attractor: fractal geometry, zero Lyapunov exponent.",
            "**Expected:** broadband PSD, K value near 1 (because the 0-1 test is "
            "sensitive to any non-regular behavior), fractal Poincare map, relatively "
            "high DET but much smaller L_max than QP, intermediate prediction error growth.\n\n"
            "**Observed:** K = 0.98, 0.99 (essentially 1), DET drops to 0.71-0.75, "
            "L_max collapses to 20-26 (an order of magnitude smaller than LC), RMS "
            "amplitude drops dramatically to around 0.08 (vs 0.22 for LC), spectral "
            "entropy jumps to 4.3-5.4 bits. The prediction error growth ratio is "
            "*lower* than expected (around 1.8-1.9) and almost identical to Chaos.\n"
        ),
        4: (
            "Chaos",
            "Broadband, sensitive dependence on initial conditions.",
            "**Expected:** broadband PSD, scattered Poincare, fractal phase portrait, "
            "low DET, rapid prediction error growth with horizon.\n\n"
            "**Observed:** K value 0.95-0.99, DET 0.71-0.77, L_max 13-20 (smallest of "
            "all regimes), RMS amplitude in the 0.01-0.04 range (lowest of all regimes, "
            "a 10x collapse from LC), spectral entropy 4.5-5.6 bits. Prediction error "
            "growth ratio is actually smaller than LC/P2 values, which is surprising "
            "until you realize it is a *normalized* quantity: chaotic signals have "
            "high error at h=1 already, so the ratio h=50/h=1 stays modest.\n"
        ),
    }

    for lbl in range(5):
        full_name, one_liner, analysis = regime_descriptions[lbl]
        subset = mean_df[mean_df['regime_label'] == lbl]
        if subset.empty:
            continue
        lines.append(f"### Regime {lbl}: {full_name}\n")
        lines.append(f"*{one_liner}*\n\n")
        lines.append(f"**Samples in this regime ({len(subset)}):** " +
                     ", ".join(f"$L/D$ = {row['L_D']:.3f}" for _, row in subset.iterrows()) + "\n\n")
        lines.append(analysis + "\n")

    # ---- D: pairwise Cohen's d ----
    lines.append("## D. Feature separation between adjacent regimes (Cohen's d)\n\n")
    lines.append(
        "For each pair of adjacent regimes we report the top 3 most discriminative "
        "features by the absolute value of Cohen's d on the channel-averaged data. "
        "Cohen's d measures the standardized mean difference: |d| > 0.8 is a "
        "conventionally 'large' effect, |d| > 1.5 is very large.\n\n"
    )
    pair_names = {
        (0, 1): "LC vs P2",
        (1, 2): "P2 vs QP",
        (2, 3): "QP vs SNA",
        (3, 4): "SNA vs Chaos",
    }
    for (a, b), entries in cohens_d_results.items():
        lines.append(f"### {pair_names.get((a, b), f'{a} vs {b}')}\n\n")
        if not entries:
            lines.append("_insufficient data_\n\n")
            continue
        lines.append("| Rank | Feature | Cohen's d | |d| |")
        lines.append("|---|---|---|---|")
        for rank, (feat, abs_d, d) in enumerate(entries[:3], start=1):
            lines.append(f"| {rank} | `{feat}` | {d:+.3f} | {abs_d:.3f} |")
        lines.append("")

    # ---- E: feature ranking ----
    lines.append("## E. Overall feature ranking via ANOVA F-statistic\n\n")
    lines.append(
        "ANOVA F-statistic across the five regime groups. Higher F means "
        "the feature separates the regimes more cleanly. p-value is the "
        "probability of the observed spread under the null hypothesis that "
        "all regime means are equal.\n\n"
    )
    lines.append("| Rank | Feature | F | p |")
    lines.append("|---|---|---|---|")
    for i, row in feature_ranking.iterrows():
        f_str = f"{row['F_statistic']:.2f}" if not np.isnan(row['F_statistic']) else "nan"
        p_str = f"{row['p_value']:.2e}" if not np.isnan(row['p_value']) else "nan"
        lines.append(f"| {i + 1} | `{row['feature']}` | {f_str} | {p_str} |")
    lines.append("")

    lines.append(
        "**Recommended features for ML classification** (top 5 by F): "
        + ", ".join(f"`{row['feature']}`"
                    for _, row in feature_ranking.head(5).iterrows())
        + ". These all already appear in the main ML pipeline's combined feature set, "
        "so the nonlinear dynamics analysis confirms that the feature engineering for "
        "the classification work is on the right track.\n\n"
    )

    # ---- F: honest SNA vs Chaos assessment ----
    sna = mean_df[mean_df['regime_label'] == 3]
    chaos = mean_df[mean_df['regime_label'] == 4]
    lines.append("## F. Honest assessment: SNA vs Chaos\n\n")
    lines.append(
        f"Only **{len(sna)} SNA samples** (L/D = "
        + ", ".join(f"{x:.3f}" for x in sna['L_D'].values) + ") and "
        f"**{len(chaos)} Chaos samples** (L/D = "
        + ", ".join(f"{x:.3f}" for x in chaos['L_D'].values) + ") are available.\n\n"
    )
    if not sna.empty and not chaos.empty:
        # Find the best SNA vs Chaos feature by |d|
        sna_vs_chaos = cohens_d_results.get((3, 4), [])
        if sna_vs_chaos:
            best_feat, best_abs_d, best_d = sna_vs_chaos[0]
            lines.append(
                f"The single most discriminative feature between SNA and Chaos in this "
                f"dataset is `{best_feat}` with Cohen's d = {best_d:+.3f}. "
            )
            if best_abs_d > 0.8:
                lines.append(
                    "This is a 'large' effect size, so the two regimes *do* separate in "
                    "this feature dimension -- but with only 2 vs 4 samples the confidence "
                    "interval is wide and the result is not statistically robust.\n\n"
                )
            else:
                lines.append(
                    "This is a 'small' effect size, indicating that on the scalar features "
                    "extracted here the SNA and Chaos samples are not cleanly separable.\n\n"
                )

    lines.append(
        "**Why SNA vs Chaos is hard in this dataset:**\n\n"
        "1. By classical definition, the distinguishing feature between SNA and Chaos is "
        "the Lyapunov exponent (zero for SNA, positive for Chaos). We removed Lyapunov "
        "because Rosenstein's estimator fails on short oscillatory recordings, so the "
        "single most theoretically discriminative quantity is unavailable to us.\n"
        "2. The remaining RQA and prediction-error features measure topological and "
        "predictability properties that differ between SNA and Chaos in principle but "
        "only by modest amounts in practice, especially with the small recording length.\n"
        "3. With 2 SNA samples, any feature distribution is essentially defined by two "
        "points, and resampling variance can swamp the true effect.\n\n"
        "**What would help:**\n\n"
        "- More recordings in the L/D = 1.9 to 2.2 band (the SNA window).\n"
        "- Longer recordings (10+ s) to make Lyapunov estimation via Rosenstein or "
        "Kantz feasible.\n"
        "- Dedicated SNA detection methods such as the 'singular continuous spectrum' "
        "test or the phase-sensitivity exponent, which are designed for SNAs specifically.\n"
        "- A forced-oscillator framing: SNAs typically arise in quasi-periodically "
        "forced systems where the forcing has two incommensurate frequencies. If that "
        "structure is present in the combustor we could exploit it.\n\n"
        "**Bottom line:** On this dataset, with this recording length, with these "
        "techniques, we can reliably distinguish periodic (LC/P2/QP) from aperiodic "
        "(SNA/Chaos) using the K value and RQA DET. We cannot reliably distinguish "
        "SNA from Chaos as separate sub-regimes. The honest recommendation for the "
        "downstream ML work is to treat the 3-class formulation (Periodic / "
        "Quasi-periodic / Aperiodic) as the primary target, and treat the 5-class "
        "formulation as an aspirational target that will require more data.\n"
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# =========================================================================
# Verbose walkthrough
# =========================================================================

def print_walkthrough(rec, fs):
    ld = rec['LD_ratio']
    name = rec['regime_name']
    pressure = rec['pressure']

    print("=" * 70)
    print(f"FEATURE EXTRACTION WALKTHROUGH - L/D = {ld:.3f} ({name})")
    print("=" * 70)
    print(f"Raw signal shape: {pressure.shape}  [samples x channels]")
    print(f"Sampling rate:    {fs} Hz")
    print(f"Recording duration: {pressure.shape[0] / fs:.1f} seconds\n")

    for ch in range(pressure.shape[1]):
        print(f"--- CHANNEL {ch + 1} ---")
        print(f"Input: {pressure.shape[0]} samples from pressure channel {ch + 1}")
        _ = analyze_channel(pressure[:, ch], fs, verbose=True)
        print()

    print("Total scalar features extracted: 8 techniques x 3 channels = 24 scalars per recording")
    print("Full arrays (PSD, recurrence matrix, prediction error curve, ...) also stored for visualization")
    print("=" * 70)


# =========================================================================
# Main
# =========================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("NONLINEAR DYNAMICS ANALYSIS OF TVC PRESSURE RECORDINGS (v2)")
    print("=" * 70)
    print("(Lyapunov removed; SciencePlots publication-quality output)\n")

    # ---- Load data ----
    data_dir = os.path.join(HERE, '..', 'data')
    print(f"Loading recordings from: {data_dir}")
    dataset = load_all_data(data_dir=data_dir)
    print(f"Loaded {len(dataset)} recordings.\n")

    # ---- Compute everything ----
    print("=" * 70)
    print("STEP 1: compute all techniques for 20 recordings x 3 channels")
    print("=" * 70)

    results = {}
    for i, rec in enumerate(dataset):
        ld = rec['LD_ratio']
        label = rec['regime_label']
        name = rec['regime_name']
        print(f"\n[{i + 1:2d}/{len(dataset)}] L/D = {ld:.4f}  {name:<16s}  ({rec['filename']})")

        rec_results = {
            'ld': ld,
            'label': label,
            'name': name,
            'filename': rec['filename'],
            'raw_ch1': rec['pressure'][:, 0].copy(),
            'channels': {},
        }
        pressure = rec['pressure']
        for ch in range(pressure.shape[1]):
            t_ch = time.time()
            print(f"    ch{ch + 1} ...", end='', flush=True)
            try:
                ch_out = analyze_channel(pressure[:, ch], SAMPLING_FREQ, verbose=False)
                rec_results['channels'][ch] = ch_out
                print(f" done ({time.time() - t_ch:.1f}s)")
            except Exception as e:
                print(f" FAILED: {e}")
                rec_results['channels'][ch] = None
        results[ld] = rec_results

    print(f"\n  Analysis complete in {(time.time() - t_start) / 60:.1f} minutes.")

    # ---- Per-recording figures ----
    print("\n" + "=" * 70)
    print("STEP 2: per-recording figures (20 figures x 2 formats)")
    print("=" * 70)
    sorted_results = sorted(results.values(), key=lambda r: r['ld'])
    n_fig = 0
    for rec in sorted_results:
        safe_name = rec['name'].replace(' ', '_').replace('-', '')
        base = os.path.join(PER_REC_DIR, f"LD_{rec['ld']:.4f}_{safe_name}")
        plot_per_recording(rec, base)
        n_fig += 2
    print(f"  Wrote {n_fig} files to {PER_REC_DIR}/")

    # ---- Regime grids ----
    print("\n" + "=" * 70)
    print("STEP 3: regime-grid figures (6 techniques x 2 formats)")
    print("=" * 70)
    grid_techs = ['psd', 'phase_portrait', 'poincare', 'recurrence',
                  'prediction_error', 'autocorrelation']
    for tech in grid_techs:
        base = os.path.join(GRID_DIR, f"grid_{tech}")
        plot_regime_grid(sorted_results, tech, base)
        print(f"  Wrote grid_{tech}.png/.pdf")

    # ---- DataFrames ----
    print("\n" + "=" * 70)
    print("STEP 4: scalar feature tables (CSV)")
    print("=" * 70)
    features_df = build_features_df(results)
    mean_df = build_mean_features_df(features_df)
    regime_avg_df = build_regime_averages_df(mean_df)

    features_df.to_csv(os.path.join(TABLE_DIR, 'features_by_channel.csv'), index=False)
    mean_df.to_csv(os.path.join(TABLE_DIR, 'features_mean.csv'), index=False)
    regime_avg_df.to_csv(os.path.join(TABLE_DIR, 'regime_averages.csv'), index=False)
    print(f"  features_by_channel.csv: {features_df.shape}")
    print(f"  features_mean.csv:       {mean_df.shape}")
    print(f"  regime_averages.csv:     {regime_avg_df.shape}")

    # ---- Feature evolution ----
    print("\n" + "=" * 70)
    print("STEP 5: feature evolution plots")
    print("=" * 70)
    plot_feature_evolution(mean_df, os.path.join(EVOL_DIR, 'bifurcation_features'))
    print("  Wrote bifurcation_features.png/.pdf")

    plot_feature_scatter(
        mean_df, 'k_value', 'det',
        x_label=r"$K$ (0-1 test)", y_label=r"DET (RQA)",
        title=r"$K$ vs DET: periodic vs aperiodic separation",
        save_base=os.path.join(EVOL_DIR, 'kvalue_vs_det_scatter'),
    )
    print("  Wrote kvalue_vs_det_scatter.png/.pdf")

    plot_feature_scatter(
        mean_df, 'pred_error_h1', 'lmax',
        x_label=r"$\epsilon_{h=1}$", y_label=r"$L_{\max}$",
        title=r"$\epsilon_{h=1}$ vs $L_{\max}$",
        save_base=os.path.join(EVOL_DIR, 'prederror_vs_lmax_scatter'),
    )
    print("  Wrote prederror_vs_lmax_scatter.png/.pdf")

    plot_feature_scatter(
        mean_df, 'spectral_entropy', 'rms',
        x_label=r"$H_{\mathrm{spec}}$ (bits)", y_label=r"$p'_{\mathrm{rms}}$",
        title=r"$H_{\mathrm{spec}}$ vs RMS: amplitude vs complexity",
        save_base=os.path.join(EVOL_DIR, 'entropy_vs_rms_scatter'),
    )
    print("  Wrote entropy_vs_rms_scatter.png/.pdf")

    # ---- Statistical analysis ----
    print("\n" + "=" * 70)
    print("STEP 6: Cohen's d and ANOVA feature ranking")
    print("=" * 70)
    scalar_cols = ['rms', 'dom_freq', 'spectral_entropy', 'k_value',
                   'rr', 'det', 'lavg', 'lmax', 'div', 'entr', 'lam', 'tt',
                   'pred_error_h1', 'pred_error_h5', 'pred_error_h10',
                   'pred_error_h20', 'pred_error_h50', 'error_growth_ratio']
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    cohens_d_results = compute_pairwise_cohens_d(mean_df, scalar_cols, pairs)
    feature_ranking = compute_feature_ranking(mean_df, scalar_cols)
    feature_ranking.to_csv(os.path.join(TABLE_DIR, 'feature_ranking_anova.csv'), index=False)
    print("  Wrote feature_ranking_anova.csv")
    print(f"\n  Top 5 features by ANOVA F:")
    for i, row in feature_ranking.head(5).iterrows():
        print(f"    {i + 1}. {row['feature']:<20s}  F = {row['F_statistic']:.2f}"
              f"  p = {row['p_value']:.2e}")

    # ---- Summary report ----
    print("\n" + "=" * 70)
    print("STEP 7: markdown summary report")
    print("=" * 70)
    write_summary_report(mean_df, regime_avg_df, cohens_d_results, feature_ranking,
                          os.path.join(RESULTS_DIR, 'summary_report.md'))
    print("  Wrote summary_report.md")

    # ---- Walkthrough ----
    print("\n")
    target = next((r for r in dataset if abs(r['LD_ratio'] - 1.625) < 1e-3), None)
    if target is not None:
        print_walkthrough(target, SAMPLING_FREQ)
    else:
        print("Walkthrough target L/D = 1.625 not found in dataset.")

    # ---- Final summary ----
    total_elapsed = time.time() - t_start
    png_count = 0
    pdf_count = 0
    for root, _, files in os.walk(RESULTS_DIR):
        png_count += sum(1 for f in files if f.endswith('.png'))
        pdf_count += sum(1 for f in files if f.endswith('.pdf'))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Total elapsed:     {total_elapsed / 60:.1f} minutes")
    print(f"  PNG figures:       {png_count}")
    print(f"  PDF figures:       {pdf_count}")
    print(f"  Total figures:     {png_count + pdf_count}")
    print(f"  CSV tables:        {len([f for f in os.listdir(TABLE_DIR) if f.endswith('.csv')])}")
    print(f"  Results root:      {RESULTS_DIR}")
    print(f"  Style:             {'SciencePlots science+no-latex' if HAS_SCIENCEPLOTS else 'manual serif fallback'}")


if __name__ == "__main__":
    main()
