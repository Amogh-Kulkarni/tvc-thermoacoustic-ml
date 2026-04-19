"""
Recurrence plots and Recurrence Quantification Analysis (RQA).

Implements:
    compute_recurrence_plot     - binary recurrence matrix from delay embedding
    compute_rqa_features        - 8 standard RQA scalar features

The RQA implementation is manual (no pyrqa/pyunicorn dependency). The
algorithms follow Marwan et al., Physics Reports 438 (2007): scan
off-diagonals for diagonal lines, scan columns for vertical lines,
collect run lengths, compute histogram-based statistics.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from existing_methods import estimate_delay_from_autocorr


# =========================================================================
# Recurrence plot construction
# =========================================================================

def _delay_embed(x, m, delay):
    """Delay-embedded signal of shape (N - (m-1)*delay, m)."""
    n = len(x) - (m - 1) * delay
    if n <= 0:
        return np.zeros((0, m))
    out = np.zeros((n, m))
    for d in range(m):
        out[:, d] = x[d * delay : d * delay + n]
    return out


def compute_recurrence_plot(x, m=3, delay=None, threshold_percentile=10, max_points=500):
    """Binary recurrence matrix.

    Subsamples the signal to max_points points, time-delay embeds in
    dimension m, computes pairwise Euclidean distances, and thresholds
    at the specified percentile.

    Returns (R, threshold) where R is an int8 (N, N) matrix.
    """
    if delay is None:
        delay = estimate_delay_from_autocorr(x)
    delay = max(1, int(delay))

    # Subsample
    step = max(1, len(x) // max_points)
    x_sub = x[::step][:max_points]
    delay_sub = max(1, delay // step)

    embedded = _delay_embed(x_sub, m, delay_sub)
    N = len(embedded)
    if N < 10:
        return np.zeros((10, 10), dtype=np.int8), 0.0

    # Pairwise distances
    D = squareform(pdist(embedded))

    # Threshold at the given percentile of nonzero distances
    nonzero = D[D > 0]
    if len(nonzero) == 0:
        threshold = 0.0
    else:
        threshold = float(np.percentile(nonzero, threshold_percentile))

    R = (D <= threshold).astype(np.int8)
    return R, threshold


# =========================================================================
# RQA feature extraction (manual)
# =========================================================================

def _runs_of_ones(arr):
    """Lengths of consecutive 1-runs in a 1D binary array."""
    if len(arr) == 0:
        return []
    arr_int = np.asarray(arr, dtype=np.int8)
    padded = np.concatenate(([0], arr_int, [0]))
    diff = np.diff(padded.astype(np.int16))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).tolist()


def _extract_diagonal_lines(R):
    """All diagonal line lengths from both triangles, excluding main diagonal."""
    N = R.shape[0]
    lines = []
    for k in range(1, N):
        lines.extend(_runs_of_ones(np.diagonal(R, offset=k)))
        lines.extend(_runs_of_ones(np.diagonal(R, offset=-k)))
    return lines


def _extract_vertical_lines(R):
    """All vertical line lengths (full columns)."""
    lines = []
    for col in range(R.shape[1]):
        lines.extend(_runs_of_ones(R[:, col]))
    return lines


def compute_rqa_features(R, l_min=2, v_min=2):
    """Compute 8 standard RQA features from binary recurrence matrix R.

    Returns a dict with keys:
        rr    - recurrence rate (fraction of 1s, main diagonal excluded)
        det   - determinism (fraction of recurrence points on diagonals >= l_min)
        l_avg - mean length of diagonal lines >= l_min
        l_max - longest diagonal line (excluding main)
        div   - divergence (1 / l_max)
        entr  - Shannon entropy of diagonal line-length distribution
        lam   - laminarity (fraction of recurrence points on verticals >= v_min)
        tt    - trapping time (mean length of vertical lines >= v_min)
    """
    N = R.shape[0]

    # Zero out main diagonal for all line analysis
    R_no_diag = R.copy()
    np.fill_diagonal(R_no_diag, 0)

    recurrence_points = int(R_no_diag.sum())
    total_offdiag = N * N - N
    rr = recurrence_points / max(total_offdiag, 1)

    # Diagonals (both triangles - matches our recurrence_points denominator)
    diag_lines = _extract_diagonal_lines(R_no_diag)
    diag_arr = np.array(diag_lines) if diag_lines else np.array([], dtype=int)
    long_diag = diag_arr[diag_arr >= l_min] if len(diag_arr) > 0 else np.array([], dtype=int)

    if len(long_diag) > 0:
        det = float(long_diag.sum()) / max(recurrence_points, 1)
        l_avg = float(long_diag.mean())
        l_max = int(long_diag.max())
        vals, counts = np.unique(long_diag, return_counts=True)
        p_l = counts / counts.sum()
        entr = float(-np.sum(p_l * np.log2(p_l + 1e-12)))
    else:
        det = 0.0
        l_avg = 0.0
        l_max = 0
        entr = 0.0

    div = (1.0 / l_max) if l_max > 0 else float('nan')

    # Verticals
    vert_lines = _extract_vertical_lines(R_no_diag)
    vert_arr = np.array(vert_lines) if vert_lines else np.array([], dtype=int)
    long_vert = vert_arr[vert_arr >= v_min] if len(vert_arr) > 0 else np.array([], dtype=int)

    if len(long_vert) > 0:
        lam = float(long_vert.sum()) / max(recurrence_points, 1)
        tt = float(long_vert.mean())
    else:
        lam = 0.0
        tt = 0.0

    return {
        'rr': rr,
        'det': det,
        'l_avg': l_avg,
        'l_max': l_max,
        'div': div,
        'entr': entr,
        'lam': lam,
        'tt': tt,
    }
