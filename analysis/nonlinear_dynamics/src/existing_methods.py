"""
Existing nonlinear dynamics techniques.

Implements:
    compute_psd                     - Welch power spectral density
    compute_dominant_frequency      - frequency of max PSD (excluding low freqs)
    compute_spectral_entropy        - Shannon entropy of normalized PSD
    compute_k_value                 - Gottwald-Melbourne 0-1 test for chaos
    compute_poincare_points         - local maxima pairs (p_n, p_{n+1})
    compute_autocorrelation         - normalized autocorrelation via FFT
    estimate_delay_from_autocorr    - first minimum of autocorrelation
    compute_phase_portrait          - 2D time-delay embedding for visualization
"""
import numpy as np
from scipy import signal as sig


# =========================================================================
# Power spectral density
# =========================================================================

def compute_psd(x, fs, nperseg=4096):
    """Welch PSD. Returns (frequencies, psd)."""
    nperseg = min(nperseg, len(x))
    f, Pxx = sig.welch(x, fs=fs, nperseg=nperseg)
    return f, Pxx


def compute_dominant_frequency(f, Pxx, min_freq=20.0):
    """Frequency of maximum PSD power, ignoring frequencies below min_freq Hz."""
    mask = f > min_freq
    if not np.any(mask):
        return float('nan')
    return float(f[mask][np.argmax(Pxx[mask])])


def compute_spectral_entropy(Pxx):
    """Shannon entropy (base 2) of the normalized PSD."""
    p = Pxx / (Pxx.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# =========================================================================
# Gottwald-Melbourne 0-1 test (FFT-vectorized for speed)
# =========================================================================

def compute_k_value(x, c_count=200, seed=42):
    """0-1 chaos test K value. K approx 0 for periodic, K approx 1 for chaos."""
    step = max(1, len(x) // 4000)
    x = x[::step].astype(np.float64)
    N = len(x)
    if N < 50:
        return float('nan')

    j = np.arange(1, N + 1, dtype=np.float64)
    n_t = max(1, N // 10)
    t = np.arange(1, n_t + 1, dtype=np.float64)
    n_arr = np.arange(1, n_t + 1, dtype=np.float64)
    Nn_arr = N - n_arr
    n_int = n_arr.astype(int)

    rng = np.random.RandomState(seed)
    c_vals = np.pi / 5 + rng.uniform(0, 1, c_count) * 3 * np.pi / 5

    mean_xsq = float(np.mean(x)) ** 2
    kcorr = np.zeros(c_count)
    N2 = 2 * N

    for its in range(c_count):
        c = c_vals[its]
        p = np.cumsum(x * np.cos(j * c))
        q = np.cumsum(x * np.sin(j * c))

        # FFT-based linear autocorrelation
        P_fft = np.fft.rfft(p, n=N2)
        R_p = np.fft.irfft(P_fft * np.conj(P_fft), n=N2)
        Q_fft = np.fft.rfft(q, n=N2)
        R_q = np.fft.irfft(Q_fft * np.conj(Q_fft), n=N2)

        # Cumulative sums of squares for shift-sum decomposition
        p2_cs = np.empty(N + 1); p2_cs[0] = 0.0
        np.cumsum(p ** 2, out=p2_cs[1:])
        q2_cs = np.empty(N + 1); q2_cs[0] = 0.0
        np.cumsum(q ** 2, out=q2_cs[1:])

        sum_p_0 = p2_cs[N - n_int]
        sum_p_n = p2_cs[N] - p2_cs[n_int]
        cross_p = R_p[n_int]
        sum_q_0 = q2_cs[N - n_int]
        sum_q_n = q2_cs[N] - q2_cs[n_int]
        cross_q = R_q[n_int]

        D = (sum_p_n + sum_p_0 - 2 * cross_p +
             sum_q_n + sum_q_0 - 2 * cross_q) / Nn_arr

        denom = 1.0 - np.cos(c)
        if abs(denom) < 1e-15:
            denom = 1e-15
        correction = mean_xsq * (1.0 - np.cos(n_arr * c)) / denom
        M = D - correction

        if np.std(M) < 1e-12:
            kcorr[its] = 0.0
        else:
            kcorr[its] = np.corrcoef(t, M)[0, 1]

    return float(np.median(kcorr))


# =========================================================================
# Poincare first return map
# =========================================================================

def compute_poincare_points(x, fs):
    """Find local maxima, return consecutive pairs (p_n, p_{n+1}) for the return map."""
    min_dist = max(1, int(fs / 500))
    peaks, _ = sig.find_peaks(x, distance=min_dist)
    if len(peaks) < 3:
        return np.array([]), np.array([])
    pk_vals = x[peaks]
    return pk_vals[:-1], pk_vals[1:]


# =========================================================================
# Autocorrelation and delay estimation
# =========================================================================

def compute_autocorrelation(x, max_lag_samples=2000):
    """FFT-based normalized autocorrelation up to max_lag_samples lags."""
    n = len(x)
    x_c = x - np.mean(x)
    fft_x = np.fft.rfft(x_c, n=2 * n)
    acf = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf = acf / (acf[0] + 1e-12)
    return acf[:min(max_lag_samples, n)]


def estimate_delay_from_autocorr(x, max_lag=500):
    """Estimate time-delay embedding lag from first minimum of autocorrelation.

    Fallbacks: first zero crossing, then max_lag // 10.
    """
    acf = compute_autocorrelation(x, max_lag_samples=max_lag)
    # First local minimum
    for i in range(1, len(acf) - 1):
        if acf[i] < acf[i - 1] and acf[i] < acf[i + 1]:
            return int(i)
    # Fallback: first zero crossing
    for i in range(1, len(acf)):
        if acf[i] < 0:
            return int(i)
    return max(1, max_lag // 10)


# =========================================================================
# Phase portrait (2D time-delay embedding)
# =========================================================================

def compute_phase_portrait(x, delay=None, fs=20000, max_points=5000):
    """2D time-delay embedding for visualization.

    Returns (x_coord, y_coord, delay) where x_coord[i] = x[i] and
    y_coord[i] = x[i + delay]. Subsampled to max_points for plotting.
    """
    if delay is None:
        delay = estimate_delay_from_autocorr(x)
    delay = max(1, int(delay))
    n = len(x) - delay
    if n <= 0:
        return np.array([]), np.array([]), delay
    idx = np.linspace(0, n - 1, min(max_points, n)).astype(int)
    return x[idx], x[idx + delay], delay
