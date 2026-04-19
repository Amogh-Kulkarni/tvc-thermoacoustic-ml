"""
Data Loading and Preprocessing for TVC Thermoacoustic Classification
=====================================================================
Handles loading .mat files, assigning regime labels based on L/D ranges
from the bifurcation diagram, and organizing data for ML training.

IMPORTANT: You will need to adjust two things in this file:
1. The DATA_DIR path to point to your .mat files
2. The regime boundaries in assign_regime_label() based on your 
   exact bifurcation diagram
"""

import os
import re
import numpy as np
from scipy.io import loadmat
import warnings

# =============================================================================
# CONFIGURATION — ADJUST THESE TO MATCH YOUR DATA
# =============================================================================

# Directory containing your .mat files
DATA_DIR = "./data"

# Sampling frequency (Hz) — from your papers: 20 kHz
SAMPLING_FREQ = 20000

# Cavity depth (mm) — fixed at 80 mm in your experiments
CAVITY_DEPTH = 80.0

# L/D values for your 20 geometries (cavity lengths from 60 to 210 mm)
# ADJUST THESE to match your exact experimental L/D values
# These are estimated — replace with your actual values
CAVITY_LENGTHS_MM = np.array([60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
                              120, 130, 140, 150, 160, 170, 180, 190, 200, 210])

# Regime labels — these are the 5 classes from the bifurcation diagram
REGIME_LABELS = {
    0: "Limit Cycle",
    1: "Period-2",
    2: "Quasi-periodic",
    3: "SNA",
    4: "Chaos"
}


def assign_regime_label(LD_ratio):
    """Assign regime label based on L/D ratio from the bifurcation diagram.
    
    IMPORTANT: These boundaries are estimated from the SoTiC 2023 and 
    ASME GT2024 papers (Figures 2 and 3). You MUST verify and adjust 
    these boundaries based on your exact bifurcation diagram at 
    Re=8000, φ=0.72.
    
    The bifurcation sequence is:
    Limit cycle → Period-2 → Quasi-periodic → SNA → Chaos
    as L/D increases from 0.75 to 2.65
    
    Args:
        LD_ratio: length-to-depth ratio of the cavity
        
    Returns:
        label: integer regime label (0-4)
    """
    # --- ADJUST THESE BOUNDARIES ---
    if LD_ratio < 1.06:
        return 0   # Limit Cycle
    elif LD_ratio < 1.25:
        return 1   # Period-2 (steeper limit cycle with higher harmonics)
    elif LD_ratio < 2.00:
        return 2   # Quasi-periodic
    elif LD_ratio < 2.20:
        return 3   # SNA (Strange Non-chaotic Attractor)
    else:
        return 4   # Chaos


def load_single_mat_file(filepath):
    """Load a single .mat file and extract pressure and time arrays.
    
    Expected structure of the .mat file:
    - p_SLPM: array of shape (40000, 3) — pressure at 3 locations
    - time: array of shape (40000, 1) — time stamps
    
    IMPORTANT: The variable names 'p_SLPM' and 'time' may differ in your 
    actual files. If loading fails, inspect the .mat file with:
        data = loadmat('your_file.mat')
        print(data.keys())
    and adjust the variable names below.
    
    Args:
        filepath: path to the .mat file
        
    Returns:
        pressure: numpy array (n_samples, 3)
        time: numpy array (n_samples,)
    """
    data = loadmat(filepath)
    
    # --- ADJUST VARIABLE NAMES IF NEEDED ---
    # Try common variable names
    pressure_keys = ['p_SLPM', 'pressure', 'p', 'P', 'p_prime']
    time_keys = ['time', 't', 'Time', 'T']
    
    pressure = None
    time_arr = None
    
    # Find pressure data
    for key in pressure_keys:
        if key in data:
            pressure = np.array(data[key], dtype=np.float64)
            break
    
    # Find time data
    for key in time_keys:
        if key in data:
            time_arr = np.array(data[key], dtype=np.float64).flatten()
            break
    
    if pressure is None:
        # Print available keys to help debug
        available = [k for k in data.keys() if not k.startswith('__')]
        raise KeyError(
            f"Could not find pressure data in {filepath}. "
            f"Available variables: {available}. "
            f"Update the pressure_keys list in load_single_mat_file()."
        )
    
    if time_arr is None:
        # If no time array, create one assuming constant sampling rate
        warnings.warn(f"No time array found in {filepath}. Creating from sampling rate.")
        time_arr = np.arange(pressure.shape[0]) / SAMPLING_FREQ
    
    # Ensure pressure is (n_samples, 3)
    if pressure.ndim == 1:
        pressure = pressure.reshape(-1, 1)
    if pressure.shape[1] > pressure.shape[0]:
        pressure = pressure.T  # Transpose if channels are in rows
    
    return pressure, time_arr


def load_all_data(data_dir=None, cavity_lengths=None):
    """Load all .mat files and organize into a dataset.
    
    This function expects .mat files in the data directory. It attempts
    to match each file to a cavity length / L/D value. 
    
    YOU WILL LIKELY NEED TO CUSTOMIZE THIS based on how your files are named.
    
    Option A: Files are named with the cavity length (e.g., 'L60.mat', 'L80.mat')
    Option B: Files are numbered sequentially (e.g., 'run_01.mat', 'run_02.mat')
    Option C: You provide an explicit mapping
    
    Args:
        data_dir: directory containing .mat files
        cavity_lengths: list of cavity lengths in mm, ordered to match file order
        
    Returns:
        dataset: list of dicts, each containing:
            - 'pressure': (n_samples, 3) array
            - 'time': (n_samples,) array  
            - 'cavity_length_mm': float
            - 'LD_ratio': float
            - 'regime_label': int
            - 'regime_name': str
            - 'filename': str
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if cavity_lengths is None:
        cavity_lengths = CAVITY_LENGTHS_MM
    
    # Find all .mat files
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")

    # Extract cavity length from filename (pattern: L_<number>_...)
    file_length_pairs = []
    for mat_file in mat_files:
        match = re.search(r'L_(\d+)', mat_file)
        if match:
            L_mm = float(match.group(1))
        else:
            warnings.warn(f"Could not extract cavity length from {mat_file}, skipping.")
            continue
        file_length_pairs.append((mat_file, L_mm))

    # Sort by cavity length for consistent ordering
    file_length_pairs.sort(key=lambda x: x[1])

    dataset = []
    for i, (mat_file, L_mm) in enumerate(file_length_pairs):
        filepath = os.path.join(data_dir, mat_file)
        
        try:
            pressure, time_arr = load_single_mat_file(filepath)
        except Exception as e:
            print(f"Warning: Could not load {mat_file}: {e}")
            continue
        
        LD = L_mm / CAVITY_DEPTH
        label = assign_regime_label(LD)
        
        dataset.append({
            'pressure': pressure,
            'time': time_arr,
            'cavity_length_mm': L_mm,
            'LD_ratio': LD,
            'regime_label': label,
            'regime_name': REGIME_LABELS[label],
            'filename': mat_file
        })
        
        print(f"  Loaded {mat_file}: L/D = {LD:.3f} -> {REGIME_LABELS[label]}")
    
    # Print class distribution
    print(f"\n--- Class Distribution ---")
    for label, name in REGIME_LABELS.items():
        count = sum(1 for d in dataset if d['regime_label'] == label)
        print(f"  {name}: {count} recordings")
    print(f"  Total: {len(dataset)} recordings")
    
    return dataset


def create_demo_data(n_recordings=20, n_samples=40000, n_channels=3, fs=20000):
    """Create synthetic demo data for testing the pipeline before real data arrives.
    
    Generates pressure signals that mimic the qualitative behavior of each regime:
    - Limit cycle: clean sinusoid at ~225 Hz
    - Period-2: two-frequency signal with subharmonic
    - Quasi-periodic: two incommensurate frequencies
    - SNA: quasi-periodic with amplitude modulation
    - Chaos: broadband noise with some spectral structure
    
    This is NOT physically accurate — it's just for testing the code pipeline.
    
    Returns:
        dataset: list of dicts in the same format as load_all_data()
    """
    t = np.arange(n_samples) / fs
    cavity_lengths = np.linspace(60, 210, n_recordings)
    
    dataset = []
    for i, L_mm in enumerate(cavity_lengths):
        LD = L_mm / CAVITY_DEPTH
        label = assign_regime_label(LD)
        
        # Generate synthetic pressure based on regime type
        f_acoustic = 225.0  # Hz, natural acoustic frequency
        
        if label == 0:  # Limit cycle — clean sinusoid
            base = 2000 * np.sin(2 * np.pi * f_acoustic * t)
            noise = 50 * np.random.randn(n_samples)
            p = base + noise
            
        elif label == 1:  # Period-2 — fundamental + subharmonic
            base = 1800 * np.sin(2 * np.pi * f_acoustic * t)
            subharm = 600 * np.sin(2 * np.pi * f_acoustic * 2 * t)
            noise = 80 * np.random.randn(n_samples)
            p = base + subharm + noise
            
        elif label == 2:  # Quasi-periodic — two incommensurate frequencies
            f2 = f_acoustic * 0.618  # Golden ratio for incommensurate
            base = 1500 * np.sin(2 * np.pi * f_acoustic * t)
            qp = 800 * np.sin(2 * np.pi * f2 * t)
            noise = 100 * np.random.randn(n_samples)
            p = base + qp + noise
            
        elif label == 3:  # SNA — quasi-periodic with irregular modulation
            f2 = f_acoustic * 0.618
            modulation = 1 + 0.5 * np.sin(2 * np.pi * 3.7 * t) * np.random.choice([-1, 1], size=n_samples)
            base = 1200 * np.sin(2 * np.pi * f_acoustic * t) * modulation
            qp = 500 * np.sin(2 * np.pi * f2 * t)
            noise = 200 * np.random.randn(n_samples)
            p = base + qp + noise
            
        else:  # Chaos — broadband with weak tonal content
            base = 500 * np.sin(2 * np.pi * f_acoustic * t)
            broadband = 800 * np.random.randn(n_samples)
            # Add some low-frequency modulation
            p = base + broadband
        
        # Create 3 channels with slight variations (simulating different locations)
        pressure = np.column_stack([
            p,
            p * 0.9 + 30 * np.random.randn(n_samples),  # Slightly different amplitude
            p * 0.8 + 50 * np.random.randn(n_samples) + 100 * np.sin(2 * np.pi * f_acoustic * t + 0.3)
        ])
        
        dataset.append({
            'pressure': pressure,
            'time': t,
            'cavity_length_mm': L_mm,
            'LD_ratio': LD,
            'regime_label': label,
            'regime_name': REGIME_LABELS[label],
            'filename': f'synthetic_L{int(L_mm)}.mat'
        })
    
    print(f"Created {n_recordings} synthetic recordings")
    print(f"\n--- Class Distribution ---")
    for label, name in REGIME_LABELS.items():
        count = sum(1 for d in dataset if d['regime_label'] == label)
        print(f"  {name}: {count} recordings")
    
    return dataset
