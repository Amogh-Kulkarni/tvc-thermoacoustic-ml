"""
Feature Extraction for TVC Thermoacoustic Regime Classification
================================================================
Extracts physics-informed features from pressure time-series data.
These features are the same quantities used in nonlinear time-series
analysis (FFT, autocorrelation, 0-1 test) but packaged as a feature
vector for ML classification.

Three categories of features:
1. Single-channel features (extracted from each of the 3 pressure channels)
2. Cross-channel features (relationships between channels)
3. Within-recording variability (std of features across windows from same recording)
"""

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform


# =============================================================================
# SINGLE-CHANNEL FEATURES
# =============================================================================

def rms_amplitude(window):
    """Root mean square of pressure fluctuations.
    
    Physical meaning: Overall oscillation strength. Your bifurcation diagram
    shows this drops ~70% from limit cycle to chaos as L/D increases.
    """
    return np.sqrt(np.mean(window**2))


def dominant_frequency(window, fs):
    """Frequency with maximum power in the FFT.
    
    Physical meaning: The acoustic mode frequency. For limit cycles,
    this is sharp at f_n ≈ 225 Hz. For chaos, the "dominant" frequency
    is less meaningful because energy is spread across many frequencies.
    """
    freqs = np.fft.rfftfreq(len(window), d=1.0/fs)
    psd = np.abs(np.fft.rfft(window))**2
    # Ignore DC component (index 0) and very low frequencies
    psd[:5] = 0
    return freqs[np.argmax(psd)]


def spectral_entropy(window, fs):
    """Shannon entropy of the normalized power spectrum.
    
    Physical meaning: How "spread out" the energy is across frequencies.
    - Limit cycle: energy at ONE frequency → LOW entropy
    - Period-2: energy at 2 frequencies → slightly higher
    - Quasi-periodic: energy at several incommensurate frequencies → moderate
    - Chaos: energy spread broadly → HIGH entropy
    
    This is arguably the single most informative feature for regime classification.
    """
    psd = np.abs(np.fft.rfft(window))**2
    # Normalize to a probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-12)
    # Remove zeros to avoid log(0)
    psd_norm = psd_norm[psd_norm > 0]
    return stats.entropy(psd_norm)


def num_spectral_peaks(window, fs, prominence_threshold=0.1):
    """Number of significant peaks in the power spectrum.
    
    Physical meaning: 
    - Limit cycle → 1 peak (f_n)
    - Period-2 → 2 peaks (f_n and f_2n, the subharmonic)
    - Quasi-periodic → multiple peaks (f_n-f, f_n, f_n+f)
    - Chaos → no clear peaks or very many small ones
    
    The prominence_threshold is relative to the maximum peak height.
    """
    psd = np.abs(np.fft.rfft(window))**2
    psd = psd / (np.max(psd) + 1e-12)  # Normalize to [0, 1]
    peaks, properties = signal.find_peaks(psd, prominence=prominence_threshold)
    return len(peaks)


def autocorrelation_decay(window, fs):
    """Rate at which autocorrelation decays from its peak.
    
    Physical meaning: How quickly the signal "forgets" its past.
    - Limit cycle: autocorrelation stays high forever (slow decay) → value near 1
    - Chaos: autocorrelation drops quickly (fast decay) → value near 0
    - SNA: intermediate behavior
    
    We measure this as the autocorrelation value at a lag of 5 acoustic periods.
    This is one of the features you already use in the SoTiC 2023 paper.
    """
    # Estimate the dominant period
    dom_freq = dominant_frequency(window, fs)
    if dom_freq < 1:  # Avoid division by zero for very low frequency
        dom_freq = 100.0
    period_samples = int(fs / dom_freq)
    
    # Compute normalized autocorrelation
    n = len(window)
    autocorr = np.correlate(window, window, mode='full')
    autocorr = autocorr[n-1:]  # Take positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-12)  # Normalize
    
    # Return autocorrelation at lag = 5 periods
    lag = min(5 * period_samples, len(autocorr) - 1)
    return autocorr[lag]


def sample_entropy(window, m=2, r_factor=0.2):
    """Approximate sample entropy using a fast subsampled approach.
    
    Physical meaning: Related to predictability.
    - Periodic signals → low sample entropy (highly predictable)
    - Chaotic signals → high sample entropy (unpredictable)
    - SNAs → intermediate (strange geometry but non-chaotic dynamics)
    
    We subsample aggressively and use vectorized operations for speed.
    """
    r = r_factor * np.std(window)
    if r < 1e-10:
        return 0.0
    
    # Aggressively subsample to ~200 points for speed
    step = max(1, len(window) // 200)
    x = window[::step]
    n = len(x)
    
    if n < m + 2:
        return 0.0
    
    # Build templates
    def count_matches(tlen):
        templates = np.array([x[i:i+tlen] for i in range(n - tlen)])
        nt = len(templates)
        count = 0
        for i in range(0, nt, 2):  # Skip every other for speed
            dists = np.max(np.abs(templates[i] - templates[i+1:]), axis=1)
            count += np.sum(dists < r)
        return count
    
    B = count_matches(m)
    A = count_matches(m + 1)
    
    if B == 0:
        return 0.0
    return -np.log((A + 1e-12) / (B + 1e-12))


def kurtosis_value(window):
    """Kurtosis of the pressure signal.
    
    Physical meaning: Measures the "tailedness" of the amplitude distribution.
    - Limit cycle (sinusoidal) → kurtosis ≈ 1.5 (for a pure sine wave)
    - Gaussian noise → kurtosis ≈ 3
    - Intermittent bursts → high kurtosis (heavy tails)
    
    This helps distinguish clean periodic oscillations from intermittent behavior.
    """
    return stats.kurtosis(window, fisher=True)  # Fisher = excess kurtosis (0 for Gaussian)


def peak_to_peak_variability(window, fs):
    """Standard deviation of successive peak amplitudes.
    
    Physical meaning: Directly related to the vortex impingement timing
    variability shown in Figure 8 of the ASME GT2024 paper.
    - Limit cycle: all peaks identical → variability ≈ 0
    - Period-2: peaks alternate between two amplitudes → moderate variability
    - Chaos: peaks vary wildly → high variability
    
    This feature has a direct connection to the ROM parameter α.
    """
    peaks, _ = signal.find_peaks(window, distance=int(fs/500))  # Minimum distance between peaks
    if len(peaks) < 3:
        return 0.0
    peak_amplitudes = window[peaks]
    return np.std(peak_amplitudes) / (np.mean(np.abs(peak_amplitudes)) + 1e-12)


# =============================================================================
# CROSS-CHANNEL FEATURES
# =============================================================================

def cross_channel_coherence(window_ch1, window_ch2, fs):
    """Magnitude-squared coherence at the dominant frequency between two channels.
    
    Physical meaning: How spatially correlated the pressure field is.
    - Limit cycle (single acoustic mode): high coherence
    - Chaos (broadband, spatially decorrelated): lower coherence
    """
    freqs, coh = signal.coherence(window_ch1, window_ch2, fs=fs, nperseg=min(256, len(window_ch1)//2))
    
    # Find coherence at dominant frequency
    dom_freq = dominant_frequency(window_ch1, fs)
    idx = np.argmin(np.abs(freqs - dom_freq))
    return coh[idx]


def cross_channel_phase(window_ch1, window_ch2, fs):
    """Phase difference between channels at the dominant frequency.
    
    Physical meaning: Encodes the acoustic mode shape.
    Different regimes may excite different spatial modes with different
    phase relationships between measurement locations.
    """
    freqs, cpsd = signal.csd(window_ch1, window_ch2, fs=fs, nperseg=min(256, len(window_ch1)//2))
    dom_freq = dominant_frequency(window_ch1, fs)
    idx = np.argmin(np.abs(freqs - dom_freq))
    return np.angle(cpsd[idx])


def cross_channel_correlation(window_ch1, window_ch2):
    """Pearson correlation coefficient between two channels.
    
    Physical meaning: Broadband spatial correlation.
    A simple measure of how similar the two pressure signals are overall.
    """
    return np.corrcoef(window_ch1, window_ch2)[0, 1]


# =============================================================================
# MASTER FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_features_single_window(window_3ch, fs):
    """Extract all features from a single 3-channel pressure window.
    
    Args:
        window_3ch: numpy array of shape (n_samples, 3), pressure data from 3 channels
        fs: sampling frequency in Hz
    
    Returns:
        feature_dict: dictionary mapping feature names to values
    """
    features = {}
    
    # --- Single-channel features for each channel ---
    for ch in range(3):
        w = window_3ch[:, ch]
        prefix = f"ch{ch+1}_"
        
        features[prefix + "rms"] = rms_amplitude(w)
        features[prefix + "dom_freq"] = dominant_frequency(w, fs)
        features[prefix + "spectral_entropy"] = spectral_entropy(w, fs)
        features[prefix + "num_peaks"] = num_spectral_peaks(w, fs)
        features[prefix + "autocorr_decay"] = autocorrelation_decay(w, fs)
        features[prefix + "sample_entropy"] = sample_entropy(w)
        features[prefix + "kurtosis"] = kurtosis_value(w)
        features[prefix + "peak_variability"] = peak_to_peak_variability(w, fs)
    
    # --- Cross-channel features (3 pairs: ch1-ch2, ch1-ch3, ch2-ch3) ---
    channel_pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in channel_pairs:
        pair_name = f"ch{i+1}_ch{j+1}_"
        features[pair_name + "coherence"] = cross_channel_coherence(
            window_3ch[:, i], window_3ch[:, j], fs
        )
        features[pair_name + "phase_diff"] = cross_channel_phase(
            window_3ch[:, i], window_3ch[:, j], fs
        )
        features[pair_name + "correlation"] = cross_channel_correlation(
            window_3ch[:, i], window_3ch[:, j]
        )
    
    return features


def extract_recording_features(pressure_3ch, fs, window_ms=50, overlap=0.5):
    """Extract features from an entire recording with windowing.
    
    This function:
    1. Slices the recording into overlapping windows
    2. Extracts features from each window
    3. Computes MEAN and STD of features across all windows
    
    The MEAN captures the typical signal characteristics.
    The STD captures within-recording variability — this is a key feature
    because limit cycles have zero variability across windows while 
    chaotic/SNA signals vary from window to window.
    
    Args:
        pressure_3ch: numpy array of shape (n_samples, 3)
        fs: sampling frequency in Hz
        window_ms: window length in milliseconds
        overlap: fractional overlap between windows (0.5 = 50%)
    
    Returns:
        recording_features: dict with mean and std of each feature
        window_features_list: list of per-window feature dicts (for deep learning later)
    """
    n_samples = pressure_3ch.shape[0]
    window_size = int(window_ms * fs / 1000)  # Convert ms to samples
    step_size = int(window_size * (1 - overlap))
    
    # Generate windows
    window_features_list = []
    start = 0
    while start + window_size <= n_samples:
        window = pressure_3ch[start:start + window_size, :]
        feat = extract_features_single_window(window, fs)
        window_features_list.append(feat)
        start += step_size
    
    if len(window_features_list) == 0:
        raise ValueError("Recording too short for the specified window size")
    
    # Compute mean and std across all windows for this recording
    feature_names = list(window_features_list[0].keys())
    recording_features = {}
    
    for name in feature_names:
        values = [wf[name] for wf in window_features_list]
        recording_features[f"mean_{name}"] = np.mean(values)
        recording_features[f"std_{name}"] = np.std(values)
    
    return recording_features, window_features_list


# =============================================================================
# RECURRENCE PLOT GENERATION (for 2D-CNN later)
# =============================================================================

def compute_recurrence_plot(window, embedding_dim=3, time_delay=None, threshold_percentile=20):
    """Compute a recurrence plot from a 1D pressure window.
    
    A recurrence plot is a 2D binary image that encodes the attractor topology.
    - Limit cycle: clean parallel diagonal lines
    - Quasi-periodic: regular torus-like texture 
    - Chaos: scattered, fragmented patterns
    
    Args:
        window: 1D pressure array
        embedding_dim: dimension for time-delay embedding
        time_delay: lag for embedding (auto-estimated if None)
        threshold_percentile: percentile of distances to use as threshold
    
    Returns:
        rp: 2D binary numpy array (the recurrence plot image)
    """
    # Subsample for computational efficiency (recurrence plots are O(n^2))
    max_points = 300  # 300x300 recurrence plot
    step = max(1, len(window) // max_points)
    x = window[::step][:max_points]
    
    # Estimate time delay from first minimum of autocorrelation if not provided
    if time_delay is None:
        autocorr = np.correlate(x, x, mode='full')[len(x)-1:]
        autocorr = autocorr / (autocorr[0] + 1e-12)
        # Find first zero crossing or minimum
        for td in range(1, len(autocorr)//4):
            if autocorr[td] < 0:
                time_delay = td
                break
        if time_delay is None:
            time_delay = 1
    
    # Time-delay embedding
    n = len(x) - (embedding_dim - 1) * time_delay
    if n < 10:
        return np.zeros((100, 100))
    
    embedded = np.zeros((n, embedding_dim))
    for d in range(embedding_dim):
        embedded[:, d] = x[d * time_delay : d * time_delay + n]
    
    # Compute distance matrix
    dist_matrix = squareform(pdist(embedded))
    
    # Threshold to create binary recurrence plot
    threshold = np.percentile(dist_matrix, threshold_percentile)
    rp = (dist_matrix < threshold).astype(np.float32)
    
    return rp
