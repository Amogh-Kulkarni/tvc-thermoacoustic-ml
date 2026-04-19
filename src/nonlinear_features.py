"""
Nonlinear Dynamics Features for TVC Thermoacoustic Regime Classification
=========================================================================
Pre-computed features from the full 2-second recordings (not windowed):
1. Gottwald-Melbourne 0-1 test for chaos (K value)
2. Poincare first return map descriptors
3. Autocorrelation features (decay rate, peak ratio)
4. Basic signal features (RMS, dominant frequency) from full recording
"""

import numpy as np
from scipy import signal
from scipy.spatial import ConvexHull


def poincare_features(pressure_1d, fs):
    """Extract features from the Poincare first return map.

    Translation of MATLAB ARFM_Return_Map function.
    Finds peaks P_max(i) and computes descriptors of the
    return map (P_max(i) vs P_max(i+1)).

    Args:
        pressure_1d: 1D pressure array (full recording)
        fs: sampling frequency in Hz

    Returns:
        dict with poincare_spread, poincare_return_corr,
        poincare_num_clusters, poincare_area
    """
    # Find peaks matching MATLAB: MinPeakDistance=10, MinPeakHeight=0
    peaks_idx, _ = signal.find_peaks(pressure_1d, distance=10, height=0)
    pks = pressure_1d[peaks_idx]

    defaults = {
        'poincare_spread': 0.0,
        'poincare_return_corr': 0.0,
        'poincare_num_clusters': 1.0,
        'poincare_area': 0.0
    }

    if len(pks) < 5:
        return defaults

    # Return map points: (P_max(i), P_max(i+1))
    x_rm = pks[:-1]
    y_rm = pks[1:]

    # Feature 1: Spread - std of peaks normalized by mean
    mean_pks = np.mean(pks)
    spread = np.std(pks) / (mean_pks + 1e-12) if mean_pks > 1e-12 else 0.0

    # Feature 2: Return correlation - Pearson between consecutive peaks
    if np.std(x_rm) < 1e-12 or np.std(y_rm) < 1e-12:
        return_corr = 1.0  # constant peaks = perfectly periodic
    else:
        return_corr = np.corrcoef(x_rm, y_rm)[0, 1]

    # Feature 3: Number of clusters in the return map
    # Sort peaks, find gaps between sorted values to identify clusters
    sorted_pks = np.sort(pks)
    diffs = np.diff(sorted_pks)
    if len(diffs) > 0 and np.median(diffs) > 1e-12:
        threshold = np.median(diffs) + 2 * np.std(diffs)
        n_clusters = float(np.sum(diffs > threshold) + 1)
    else:
        n_clusters = 1.0
    n_clusters = min(n_clusters, 10.0)

    # Feature 4: Convex hull area of return map, normalized
    area = 0.0
    if len(x_rm) >= 3:
        try:
            points = np.column_stack([x_rm, y_rm])
            hull = ConvexHull(points)
            pk_range = np.max(pks) - np.min(pks)
            if pk_range > 1e-12:
                area = hull.volume / (pk_range ** 2)
        except Exception:
            area = 0.0

    return {
        'poincare_spread': spread,
        'poincare_return_corr': return_corr,
        'poincare_num_clusters': n_clusters,
        'poincare_area': area
    }


def z1_test_k_value(pressure_1d, n_c=200, seed=42):
    """Gottwald-Melbourne 0-1 test for chaos.

    Translation of MATLAB z1test function.
    K ~ 0: periodic/quasi-periodic (non-chaotic)
    K ~ 1: chaotic
    Intermediate values: SNAs

    Uses FFT-based mean-square-displacement for fast computation.
    Signal is subsampled to ~4000 points as recommended for oversampled data.

    Args:
        pressure_1d: 1D pressure array (full recording, e.g. 40000 points)
        n_c: number of random c values (default 200)
        seed: random seed for reproducibility

    Returns:
        kmedian: median K value across all c values
    """
    # Subsample to ~4000 points for speed
    step = max(1, len(pressure_1d) // 4000)
    x = pressure_1d[::step].astype(np.float64)
    N = len(x)

    j = np.arange(1, N + 1, dtype=np.float64)
    n_t = max(1, N // 10)
    t = np.arange(1, n_t + 1, dtype=np.float64)
    n_arr = np.arange(1, n_t + 1, dtype=np.float64)
    Nn_arr = N - n_arr  # (N-1, N-2, ..., N-n_t)

    # Random c values in [pi/5, 4*pi/5]
    rng = np.random.RandomState(seed)
    rand_vals = rng.uniform(0, 1, n_c)
    c_vals = np.pi / 5 + rand_vals * 3 * np.pi / 5

    mean_x_sq = np.mean(x) ** 2
    kcorr = np.zeros(n_c)
    N2 = 2 * N
    n_int = n_arr.astype(int)

    for its in range(n_c):
        c = c_vals[its]

        # Cumulative sums (translation variables p, q)
        p = np.cumsum(x * np.cos(j * c))
        q = np.cumsum(x * np.sin(j * c))

        # FFT-based autocorrelation for vectorized MSD computation
        P_fft = np.fft.rfft(p, n=N2)
        R_p = np.fft.irfft(P_fft * np.conj(P_fft), n=N2)

        Q_fft = np.fft.rfft(q, n=N2)
        R_q = np.fft.irfft(Q_fft * np.conj(Q_fft), n=N2)

        # Cumulative sums of squared values for shift-sum decomposition
        p2_cs = np.empty(N + 1)
        p2_cs[0] = 0.0
        np.cumsum(p ** 2, out=p2_cs[1:])

        q2_cs = np.empty(N + 1)
        q2_cs[0] = 0.0
        np.cumsum(q ** 2, out=q2_cs[1:])

        # Vectorized MSD for all lags n = 1..n_t
        # D(n) = (1/(N-n)) * sum_{k=0}^{N-n-1} [(p[k+n]-p[k])^2 + (q[k+n]-q[k])^2]
        #       = (1/(N-n)) * [sum p[k+n]^2 + sum p[k]^2 - 2*R_p[n] + same for q]
        sum_p_0 = p2_cs[N - n_int]           # sum p[k]^2 for k=0..N-n-1
        sum_p_n = p2_cs[N] - p2_cs[n_int]    # sum p[k]^2 for k=n..N-1
        cross_p = R_p[n_int]

        sum_q_0 = q2_cs[N - n_int]
        sum_q_n = q2_cs[N] - q2_cs[n_int]
        cross_q = R_q[n_int]

        D = (sum_p_n + sum_p_0 - 2 * cross_p +
             sum_q_n + sum_q_0 - 2 * cross_q) / Nn_arr

        # Correction term from the 0-1 test theory
        denom = 1.0 - np.cos(c)
        if abs(denom) < 1e-15:
            denom = 1e-15
        correction = mean_x_sq * (1.0 - np.cos(n_arr * c)) / denom

        M = D - correction

        # Pearson correlation between lag index t and modified MSD M
        if np.std(M) < 1e-12:
            kcorr[its] = 0.0
        else:
            kcorr[its] = np.corrcoef(t, M)[0, 1]

    return float(np.median(kcorr))


def autocorrelation_features(pressure_1d, fs):
    """Autocorrelation-based features from the full recording.

    Args:
        pressure_1d: 1D pressure array (full recording)
        fs: sampling frequency in Hz

    Returns:
        dict with autocorr_decay_10 and autocorr_peak_ratio
    """
    # Dominant frequency from full recording
    freqs = np.fft.rfftfreq(len(pressure_1d), d=1.0 / fs)
    psd = np.abs(np.fft.rfft(pressure_1d)) ** 2
    psd[:5] = 0  # Ignore DC and very low frequencies
    dom_freq = freqs[np.argmax(psd)]
    if dom_freq < 1:
        dom_freq = 100.0
    period_samples = int(fs / dom_freq)

    # FFT-based normalized autocorrelation (fast O(N log N))
    n = len(pressure_1d)
    x = pressure_1d - np.mean(pressure_1d)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf = acf / (acf[0] + 1e-12)

    # Feature 1: Autocorrelation at lag = 10 acoustic periods
    lag_10 = min(10 * period_samples, n - 1)
    decay_10 = float(acf[lag_10])

    # Feature 2: Ratio of 2nd autocorrelation peak to 1st peak
    min_dist = max(1, period_samples // 2)
    acf_peaks_idx, _ = signal.find_peaks(acf[1:], distance=min_dist)
    acf_peaks_idx = acf_peaks_idx + 1  # Adjust for skipping lag 0

    if len(acf_peaks_idx) >= 2:
        p1 = acf[acf_peaks_idx[0]]
        p2 = acf[acf_peaks_idx[1]]
        peak_ratio = float(p2 / p1) if abs(p1) > 1e-12 else 0.0
    else:
        peak_ratio = 0.0

    return {
        'autocorr_decay_10': decay_10,
        'autocorr_peak_ratio': peak_ratio
    }


def compute_all_nonlinear_features(pressure_3ch, fs):
    """Compute all nonlinear dynamics features from a 3-channel recording.

    Features are computed from the FULL recording (not windowed).
    Per channel: z1_K, 4 Poincare features, 2 autocorrelation features,
    RMS, dominant frequency = 9 features x 3 channels = 27 total.

    Args:
        pressure_3ch: numpy array of shape (n_samples, n_channels)
        fs: sampling frequency in Hz

    Returns:
        feature_dict: dictionary mapping feature names to values
    """
    n_channels = pressure_3ch.shape[1]
    features = {}

    for ch in range(n_channels):
        x = pressure_3ch[:, ch]
        prefix = f"ch{ch + 1}_"

        # Basic features from full recording
        features[prefix + "nl_rms"] = float(np.sqrt(np.mean(x ** 2)))
        f = np.fft.rfftfreq(len(x), d=1.0 / fs)
        psd_full = np.abs(np.fft.rfft(x)) ** 2
        psd_full[:5] = 0
        features[prefix + "nl_dom_freq"] = float(f[np.argmax(psd_full)])

        # 0-1 test K value
        features[prefix + "z1_K"] = z1_test_k_value(x)

        # Poincare return map features
        poinc = poincare_features(x, fs)
        for key, val in poinc.items():
            features[prefix + key] = val

        # Autocorrelation features
        acf_feats = autocorrelation_features(x, fs)
        for key, val in acf_feats.items():
            features[prefix + key] = val

    return features
