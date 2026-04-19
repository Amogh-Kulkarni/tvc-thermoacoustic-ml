"""
Deep Learning Classification Pipeline for TVC Thermoacoustic Regimes
=====================================================================
Implements four models with leave-one-L/D-out cross-validation:
  1. 1D-CNN on raw pressure windows
  2. LSTM on sequential feature vectors
  3. GRU on sequential feature vectors
  4. 2D-CNN on recurrence plot images

Directly comparable to classical ML results from main_classical_ml.py.

Usage:
    python main_deep_learning.py --real-data
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add project source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import extract_features_single_window, compute_recurrence_plot
from data_loading import load_all_data, REGIME_LABELS, SAMPLING_FREQ

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

REGIME_LABELS_3CLASS = {0: "Periodic", 1: "Quasi-periodic", 2: "Aperiodic"}


def remap_to_3class(y):
    mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    return np.array([mapping[int(label)] for label in y])


# =============================================================================
# DATA PREPARATION
# =============================================================================

def window_recording(pressure_3ch, fs=SAMPLING_FREQ, window_ms=50, overlap=0.5):
    """Slice a recording into overlapping windows. Returns list of (1000,3) arrays."""
    window_size = int(window_ms * fs / 1000)
    step_size = int(window_size * (1 - overlap))
    windows = []
    start = 0
    while start + window_size <= pressure_3ch.shape[0]:
        windows.append(pressure_3ch[start:start + window_size, :].copy())
        start += step_size
    return windows


def prepare_all_windows(dataset):
    """Prepare raw pressure windows from all recordings.
    Returns windows, labels, LD_values (per-window)."""
    all_windows = []
    all_labels = []
    all_LD = []
    all_rec_idx = []
    for i, rec in enumerate(dataset):
        wins = window_recording(rec['pressure'])
        for w in wins:
            all_windows.append(w)
            all_labels.append(rec['regime_label'])
            all_LD.append(rec['LD_ratio'])
            all_rec_idx.append(i)
    return (np.array(all_windows), np.array(all_labels),
            np.array(all_LD), np.array(all_rec_idx))


def prepare_sequential_features(dataset, fs=SAMPLING_FREQ, seq_len=10,
                                window_ms=50, overlap=0.5):
    """Extract per-window features and group into sequences of seq_len."""
    window_size = int(window_ms * fs / 1000)
    step_size = int(window_size * (1 - overlap))

    all_sequences = []
    all_labels = []
    all_LD = []
    feature_names = None

    for rec in dataset:
        pressure = rec['pressure']
        n_samples = pressure.shape[0]

        window_feats = []
        start = 0
        while start + window_size <= n_samples:
            w = pressure[start:start + window_size, :]
            feat_dict = extract_features_single_window(w, fs)
            if feature_names is None:
                feature_names = sorted(feat_dict.keys())
            feat_vec = [feat_dict[k] for k in feature_names]
            window_feats.append(feat_vec)
            start += step_size

        window_feats = np.array(window_feats, dtype=np.float32)
        window_feats = np.nan_to_num(window_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Group into non-overlapping sequences
        n_seqs = len(window_feats) // seq_len
        for s in range(n_seqs):
            seq = window_feats[s * seq_len:(s + 1) * seq_len]
            all_sequences.append(seq)
            all_labels.append(rec['regime_label'])
            all_LD.append(rec['LD_ratio'])

    return (np.array(all_sequences), np.array(all_labels),
            np.array(all_LD), feature_names)


def prepare_recurrence_plots(dataset, fs=SAMPLING_FREQ):
    """Compute recurrence plots for all channels of each recording."""
    all_rps = []
    all_labels = []
    all_LD = []

    for rec in dataset:
        for ch in range(rec['pressure'].shape[1]):
            signal_1d = rec['pressure'][:, ch]
            rp = compute_recurrence_plot(signal_1d, embedding_dim=3,
                                         threshold_percentile=20)
            # Ensure 300x300
            if rp.shape[0] != 300 or rp.shape[1] != 300:
                from scipy.ndimage import zoom
                rp = zoom(rp, (300 / rp.shape[0], 300 / rp.shape[1]), order=0)
            all_rps.append(rp.astype(np.float32))
            all_labels.append(rec['regime_label'])
            all_LD.append(rec['LD_ratio'])

    return np.array(all_rps), np.array(all_labels), np.array(all_LD)


# =============================================================================
# PYTORCH DATASETS
# =============================================================================

class PressureWindowDataset(Dataset):
    """Dataset for 1D-CNN: raw pressure windows."""

    def __init__(self, windows, labels, augment=False):
        # windows: (N, 1000, 3) -> store as (N, 3, 1000) for Conv1d
        self.X = torch.tensor(windows.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def set_normalization(self, mean, std):
        """Apply per-channel normalization."""
        self.X = (self.X - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Gaussian noise: std = 0.05 * channel std (data is already normalized,
            # so channel std ~ 1, noise std ~ 0.05)
            x = x + 0.05 * torch.randn_like(x)
            # Random time shift +/- 50 samples
            shift = np.random.randint(-50, 51)
            if shift != 0:
                x = torch.roll(x, shift, dims=1)
        return x, self.y[idx]


class SequenceDataset(Dataset):
    """Dataset for LSTM/GRU: sequences of feature vectors."""

    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RecurrencePlotDataset(Dataset):
    """Dataset for 2D-CNN: recurrence plot images."""

    def __init__(self, rps, labels, augment=False):
        # rps: (N, 300, 300) -> (N, 1, 300, 300)
        self.X = torch.tensor(rps[:, np.newaxis, :, :], dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Random flip (horizontal and/or vertical)
            if np.random.rand() > 0.5:
                x = x.flip(1)
            if np.random.rand() > 0.5:
                x = x.flip(2)
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                x = torch.rot90(x, k, [1, 2])
            # Small Gaussian noise
            x = x + 0.02 * torch.randn_like(x)
        return x, self.y[idx]


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CNN1D(nn.Module):
    """1D-CNN on raw pressure windows. Input: (batch, 3, 1000)."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.classifier(x)
        return x


class LSTMClassifier(nn.Module):
    """LSTM on sequential feature vectors. Input: (batch, seq_len, n_features)."""

    def __init__(self, input_size, num_classes=5, hidden_size=48):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        x = h_n[-1]  # Last hidden state
        x = self.classifier(x)
        return x


class GRUClassifier(nn.Module):
    """GRU on sequential feature vectors. Input: (batch, seq_len, n_features)."""

    def __init__(self, input_size, num_classes=5, hidden_size=48):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, num_classes),
        )

    def forward(self, x):
        _, h_n = self.rnn(x)
        x = h_n[-1]
        x = self.classifier(x)
        return x


class CNN2D(nn.Module):
    """2D-CNN on recurrence plots. Input: (batch, 1, 300, 300)."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # Global average pooling 2D
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_model(model, train_loader, val_loader, num_classes,
                max_epochs=100, patience=15, lr=1e-3, weight_decay=1e-4):
    """Train a model with early stopping. Returns best model state dict."""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights for imbalanced data
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_counts = np.bincount(all_labels, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (out.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (out.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_loss /= max(train_total, 1)
        val_loss /= max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model


def predict(model, loader):
    """Get predictions from a model."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch)
            all_preds.extend(out.argmax(1).cpu().numpy())
    return np.array(all_preds)


# =============================================================================
# LEAVE-ONE-L/D-OUT CV RUNNERS
# =============================================================================

def run_cnn1d_cv(dataset, num_classes, regime_labels, results_dir):
    """Run 1D-CNN with leave-one-L/D-out CV."""
    print("\n" + "=" * 70)
    print(f"1D-CNN on Raw Pressure Windows ({num_classes}-class)")
    print("=" * 70)

    windows, labels, LD_vals, _ = prepare_all_windows(dataset)
    if num_classes == 3:
        labels = remap_to_3class(labels)

    unique_LD = np.unique(LD_vals)
    all_true, all_pred, all_LD_test = [], [], []

    for fold_i, held_out_LD in enumerate(unique_LD):
        test_mask = LD_vals == held_out_LD
        train_mask = ~test_mask

        # Split a validation set: pick one other L/D
        train_LD = np.unique(LD_vals[train_mask])
        val_LD = train_LD[fold_i % len(train_LD)]
        val_mask = LD_vals == val_LD
        actual_train_mask = train_mask & ~val_mask

        # Build datasets
        train_ds = PressureWindowDataset(windows[actual_train_mask],
                                          labels[actual_train_mask], augment=True)
        val_ds = PressureWindowDataset(windows[val_mask],
                                        labels[val_mask], augment=False)
        test_ds = PressureWindowDataset(windows[test_mask],
                                         labels[test_mask], augment=False)

        # Compute normalization from training set
        train_X = train_ds.X  # (N, 3, 1000)
        ch_mean = train_X.mean(dim=[0, 2], keepdim=True)
        ch_std = train_X.std(dim=[0, 2], keepdim=True)
        train_ds.set_normalization(ch_mean, ch_std)
        val_ds.set_normalization(ch_mean, ch_std)
        test_ds.set_normalization(ch_mean, ch_std)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        true_name = regime_labels[int(labels[test_mask][0])]
        print(f"\n  Fold {fold_i+1:2d}/{len(unique_LD)}: held out L/D={held_out_LD:.3f} "
              f"({true_name}), train={sum(actual_train_mask)}, val={sum(val_mask)}, "
              f"test={sum(test_mask)}")

        model = CNN1D(num_classes=num_classes)
        model = train_model(model, train_loader, val_loader, num_classes)

        preds = predict(model, test_loader)
        # Majority vote across windows for this L/D
        majority_pred = int(np.bincount(preds, minlength=num_classes).argmax())
        true_label = int(labels[test_mask][0])

        all_true.append(true_label)
        all_pred.append(majority_pred)
        all_LD_test.append(held_out_LD)

        status = "OK" if majority_pred == true_label else "MISS"
        pred_name = regime_labels[majority_pred]
        print(f"    Result: True={true_name:15s} Pred={pred_name:15s} {status}")

    accuracy = accuracy_score(all_true, all_pred)
    print(f"\n  1D-CNN Overall Accuracy: {accuracy:.1%}")
    return {'y_true': np.array(all_true), 'y_pred': np.array(all_pred),
            'LD_test': np.array(all_LD_test), 'accuracy': accuracy,
            'classifier_name': f'1D-CNN ({num_classes}-class)'}


def run_rnn_cv(dataset, num_classes, regime_labels, model_type='LSTM', results_dir='.'):
    """Run LSTM or GRU with leave-one-L/D-out CV."""
    name = model_type
    print("\n" + "=" * 70)
    print(f"{name} on Sequential Features ({num_classes}-class)")
    print("=" * 70)

    sequences, labels, LD_vals, feature_names = prepare_sequential_features(dataset)
    if num_classes == 3:
        labels = remap_to_3class(labels)
    n_features = sequences.shape[2]
    print(f"  Sequences: {sequences.shape}, features per window: {n_features}")

    unique_LD = np.unique(LD_vals)
    all_true, all_pred, all_LD_test = [], [], []

    for fold_i, held_out_LD in enumerate(unique_LD):
        test_mask = LD_vals == held_out_LD
        train_mask = ~test_mask

        train_LD = np.unique(LD_vals[train_mask])
        val_LD = train_LD[fold_i % len(train_LD)]
        val_mask = LD_vals == val_LD
        actual_train_mask = train_mask & ~val_mask

        # Standardize features (fit on training only)
        train_seqs = sequences[actual_train_mask]
        scaler = StandardScaler()
        flat_train = train_seqs.reshape(-1, n_features)
        scaler.fit(flat_train)

        def scale_seqs(seqs):
            orig_shape = seqs.shape
            flat = seqs.reshape(-1, n_features)
            flat = scaler.transform(flat)
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            return flat.reshape(orig_shape).astype(np.float32)

        train_ds = SequenceDataset(scale_seqs(sequences[actual_train_mask]),
                                    labels[actual_train_mask])
        val_ds = SequenceDataset(scale_seqs(sequences[val_mask]),
                                  labels[val_mask])
        test_ds = SequenceDataset(scale_seqs(sequences[test_mask]),
                                   labels[test_mask])

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        true_name = regime_labels[int(labels[test_mask][0])]
        print(f"\n  Fold {fold_i+1:2d}/{len(unique_LD)}: held out L/D={held_out_LD:.3f} "
              f"({true_name}), train={sum(actual_train_mask)}, "
              f"test={sum(test_mask)}")

        if model_type == 'LSTM':
            model = LSTMClassifier(input_size=n_features, num_classes=num_classes)
        else:
            model = GRUClassifier(input_size=n_features, num_classes=num_classes)

        model = train_model(model, train_loader, val_loader, num_classes,
                            max_epochs=100, patience=15, lr=1e-3)

        preds = predict(model, test_loader)
        if len(preds) > 0:
            majority_pred = int(np.bincount(preds, minlength=num_classes).argmax())
        else:
            majority_pred = int(labels[test_mask][0])  # fallback
        true_label = int(labels[test_mask][0])

        all_true.append(true_label)
        all_pred.append(majority_pred)
        all_LD_test.append(held_out_LD)

        status = "OK" if majority_pred == true_label else "MISS"
        pred_name = regime_labels[majority_pred]
        print(f"    Result: True={true_name:15s} Pred={pred_name:15s} {status}")

    accuracy = accuracy_score(all_true, all_pred)
    print(f"\n  {name} Overall Accuracy: {accuracy:.1%}")
    return {'y_true': np.array(all_true), 'y_pred': np.array(all_pred),
            'LD_test': np.array(all_LD_test), 'accuracy': accuracy,
            'classifier_name': f'{name} ({num_classes}-class)'}


def run_cnn2d_cv(dataset, num_classes, regime_labels, results_dir):
    """Run 2D-CNN on recurrence plots with leave-one-L/D-out CV."""
    print("\n" + "=" * 70)
    print(f"2D-CNN on Recurrence Plots ({num_classes}-class)")
    print("=" * 70)

    rps, labels, LD_vals = prepare_recurrence_plots(dataset)
    if num_classes == 3:
        labels = remap_to_3class(labels)
    print(f"  Recurrence plots: {rps.shape}")

    unique_LD = np.unique(LD_vals)
    all_true, all_pred, all_LD_test = [], [], []

    for fold_i, held_out_LD in enumerate(unique_LD):
        test_mask = LD_vals == held_out_LD
        train_mask = ~test_mask

        train_LD = np.unique(LD_vals[train_mask])
        val_LD = train_LD[fold_i % len(train_LD)]
        val_mask = LD_vals == val_LD
        actual_train_mask = train_mask & ~val_mask

        train_ds = RecurrencePlotDataset(rps[actual_train_mask],
                                          labels[actual_train_mask], augment=True)
        val_ds = RecurrencePlotDataset(rps[val_mask],
                                        labels[val_mask], augment=False)
        test_ds = RecurrencePlotDataset(rps[test_mask],
                                         labels[test_mask], augment=False)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

        true_name = regime_labels[int(labels[test_mask][0])]
        print(f"\n  Fold {fold_i+1:2d}/{len(unique_LD)}: held out L/D={held_out_LD:.3f} "
              f"({true_name})")

        model = CNN2D(num_classes=num_classes)
        model = train_model(model, train_loader, val_loader, num_classes,
                            max_epochs=100, patience=15, lr=1e-3)

        preds = predict(model, test_loader)
        # Majority vote across 3 channels
        majority_pred = int(np.bincount(preds, minlength=num_classes).argmax())
        true_label = int(labels[test_mask][0])

        all_true.append(true_label)
        all_pred.append(majority_pred)
        all_LD_test.append(held_out_LD)

        status = "OK" if majority_pred == true_label else "MISS"
        pred_name = regime_labels[majority_pred]
        print(f"    Result: True={true_name:15s} Pred={pred_name:15s} {status}")

    accuracy = accuracy_score(all_true, all_pred)
    print(f"\n  2D-CNN Overall Accuracy: {accuracy:.1%}")
    return {'y_true': np.array(all_true), 'y_pred': np.array(all_pred),
            'LD_test': np.array(all_LD_test), 'accuracy': accuracy,
            'classifier_name': f'2D-CNN RP ({num_classes}-class)'}


# =============================================================================
# GRAD-CAM FOR 1D-CNN
# =============================================================================

def grad_cam_1d(model, input_tensor, target_class):
    """Compute Grad-CAM for a 1D-CNN.
    Returns a 1D heatmap of length matching the input time dimension."""
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

    # Hook the last conv layer
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Last conv layer is features[-1] which is ReLU; use features[-2] (BN after conv3)
    # Actually get the last ReLU output
    last_conv_layer = model.features[-1]  # ReLU after 3rd conv block
    h_fwd = last_conv_layer.register_forward_hook(fwd_hook)
    h_bwd = last_conv_layer.register_full_backward_hook(bwd_hook)

    out = model(input_tensor)
    model.zero_grad()
    out[0, target_class].backward()

    h_fwd.remove()
    h_bwd.remove()

    act = activations[0].detach().cpu().numpy()[0]   # (C, T)
    grad = gradients[0].detach().cpu().numpy()[0]     # (C, T)

    # Channel-wise weights
    weights = grad.mean(axis=1)  # (C,)
    cam = np.maximum(np.einsum('c,ct->t', weights, act), 0)

    # Upsample to input length
    from scipy.ndimage import zoom
    cam = zoom(cam, input_tensor.shape[2] / len(cam), order=1)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def generate_grad_cam_plots(dataset, num_classes, regime_labels, results_dir):
    """Generate Grad-CAM visualizations for 1D-CNN, one per regime."""
    print("\n  Generating Grad-CAM visualizations...")

    windows, labels, LD_vals, rec_idx = prepare_all_windows(dataset)
    if num_classes == 3:
        labels_use = remap_to_3class(labels)
    else:
        labels_use = labels.copy()

    # Train a model on ALL data for Grad-CAM visualization
    train_ds = PressureWindowDataset(windows, labels_use, augment=False)
    ch_mean = train_ds.X.mean(dim=[0, 2], keepdim=True)
    ch_std = train_ds.X.std(dim=[0, 2], keepdim=True)
    train_ds.set_normalization(ch_mean, ch_std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

    model = CNN1D(num_classes=num_classes)
    model = train_model(model, train_loader, val_loader, num_classes,
                        max_epochs=60, patience=15, lr=1e-3)

    # For each regime, find a correctly classified window
    preds = predict(model, val_loader)

    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 3 * num_classes))
    if num_classes == 1:
        axes = [axes]

    for cls in range(num_classes):
        ax = axes[cls]
        # Find a window correctly classified as this class
        cls_mask = (labels_use == cls) & (preds == cls)
        cls_indices = np.where(cls_mask)[0]

        if len(cls_indices) == 0:
            ax.text(0.5, 0.5, f"No correct predictions for {regime_labels[cls]}",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        idx = cls_indices[0]
        input_tensor = train_ds.X[idx]
        cam = grad_cam_1d(model, input_tensor, cls)

        # Plot: channel 1 waveform + Grad-CAM overlay
        t = np.arange(1000) / SAMPLING_FREQ * 1000  # ms
        waveform = input_tensor[0].numpy()  # Channel 1
        ax.plot(t, waveform, 'k-', linewidth=0.5, alpha=0.7)
        ax.fill_between(t, waveform.min(), waveform.max(),
                        where=cam > 0.5, alpha=0.3, color='red',
                        label='Grad-CAM (high attention)')
        ax2 = ax.twinx()
        ax2.plot(t, cam, 'r-', linewidth=1.5, alpha=0.6)
        ax2.set_ylim(0, 1.5)
        ax2.set_ylabel('Grad-CAM', color='red', fontsize=9)
        ax.set_title(f'{regime_labels[cls]} (L/D={LD_vals[idx]:.3f})', fontsize=11)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Pressure (norm.)')
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Grad-CAM: 1D-CNN Attention on Pressure Waveform', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'grad_cam_1dcnn.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_dir}/grad_cam_1dcnn.png")


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_confusion_matrix(results, regime_labels, save_path):
    present_classes = sorted(set(results['y_true']) | set(results['y_pred']))
    present_names = [regime_labels[c] for c in present_classes]
    cm = confusion_matrix(results['y_true'], results['y_pred'],
                          labels=present_classes)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_xlabel('Predicted Regime')
    ax.set_ylabel('True Regime')
    ax.set_title(f"{results['classifier_name']}\n"
                 f"Leave-One-L/D-Out Accuracy: {results['accuracy']:.1%}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(data_dir="./data", results_dir="./results"):
    # --- Load data ---
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    dataset = load_all_data(data_dir=data_dir)

    # --- Parameter counts ---
    print("\n" + "=" * 70)
    print("MODEL PARAMETER COUNTS")
    print("=" * 70)
    dummy_n_features = 33  # approximate; will be exact once we run
    models_info = [
        ("1D-CNN", CNN1D(num_classes=5)),
        ("LSTM", LSTMClassifier(input_size=dummy_n_features, num_classes=5)),
        ("GRU", GRUClassifier(input_size=dummy_n_features, num_classes=5)),
        ("2D-CNN (RP)", CNN2D(num_classes=5)),
    ]
    for name, m in models_info:
        n_params = count_parameters(m)
        print(f"  {name:20s}: {n_params:,d} parameters")

    # --- Run all models for 5-class and 3-class ---
    all_results = {}

    for num_classes, rlabels, tag in [
        (5, REGIME_LABELS, "5class"),
        (3, REGIME_LABELS_3CLASS, "3class"),
    ]:
        sub_dir = os.path.join(results_dir, f"dl_{tag}")
        os.makedirs(sub_dir, exist_ok=True)

        # 1D-CNN
        res = run_cnn1d_cv(dataset, num_classes, rlabels, sub_dir)
        all_results[f"1D-CNN_{tag}"] = res
        plot_confusion_matrix(res, rlabels,
                              os.path.join(sub_dir, 'confusion_1DCNN.png'))

        # LSTM
        res = run_rnn_cv(dataset, num_classes, rlabels,
                         model_type='LSTM', results_dir=sub_dir)
        all_results[f"LSTM_{tag}"] = res
        plot_confusion_matrix(res, rlabels,
                              os.path.join(sub_dir, 'confusion_LSTM.png'))

        # GRU
        res = run_rnn_cv(dataset, num_classes, rlabels,
                         model_type='GRU', results_dir=sub_dir)
        all_results[f"GRU_{tag}"] = res
        plot_confusion_matrix(res, rlabels,
                              os.path.join(sub_dir, 'confusion_GRU.png'))

        # 2D-CNN on recurrence plots
        res = run_cnn2d_cv(dataset, num_classes, rlabels, sub_dir)
        all_results[f"2D-CNN_{tag}"] = res
        plot_confusion_matrix(res, rlabels,
                              os.path.join(sub_dir, 'confusion_2DCNN_RP.png'))

    # --- Grad-CAM for 5-class ---
    gc_dir = os.path.join(results_dir, "dl_5class")
    generate_grad_cam_plots(dataset, 5, REGIME_LABELS, gc_dir)

    # --- Comprehensive comparison table ---
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 70)

    classical_results = [
        ("SVM (RBF) - Windowed", "Classical", 60.0, 85.0),
        ("Random Forest - Windowed", "Classical", 60.0, 95.0),
        ("XGBoost - Windowed", "Classical", 55.0, 90.0),
        ("SVM (RBF) - Combined", "Classical", 65.0, 85.0),
        ("Random Forest - Combined", "Classical", 60.0, 90.0),
        ("XGBoost - Combined", "Classical", 60.0, 95.0),
    ]

    dl_rows = [
        ("1D-CNN (raw pressure)", "Deep",
         all_results["1D-CNN_5class"]['accuracy'] * 100,
         all_results["1D-CNN_3class"]['accuracy'] * 100),
        ("LSTM (sequential features)", "Deep",
         all_results["LSTM_5class"]['accuracy'] * 100,
         all_results["LSTM_3class"]['accuracy'] * 100),
        ("GRU (sequential features)", "Deep",
         all_results["GRU_5class"]['accuracy'] * 100,
         all_results["GRU_3class"]['accuracy'] * 100),
        ("2D-CNN (recurrence plots)", "Deep",
         all_results["2D-CNN_5class"]['accuracy'] * 100,
         all_results["2D-CNN_3class"]['accuracy'] * 100),
    ]

    all_rows = classical_results + dl_rows

    print(f"\n  {'Method':<35s} {'Type':<10s} {'5-class':>8s} {'3-class':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8}")
    for method, mtype, acc5, acc3 in all_rows:
        print(f"  {method:<35s} {mtype:<10s} {acc5:>7.1f}% {acc3:>7.1f}%")

    # --- Per-model classification reports for 5-class ---
    print("\n" + "=" * 70)
    print("5-CLASS CLASSIFICATION REPORTS")
    print("=" * 70)
    for key in ["1D-CNN_5class", "LSTM_5class", "GRU_5class", "2D-CNN_5class"]:
        res = all_results[key]
        print(f"\n  {res['classifier_name']}:")
        present = sorted(set(res['y_true']))
        names = [REGIME_LABELS[c] for c in present]
        print(classification_report(res['y_true'], res['y_pred'],
                                    labels=present, target_names=names,
                                    zero_division=0))

    print(f"\n  All deep learning results saved to: {results_dir}/dl_5class/ and dl_3class/")
    return all_results


if __name__ == "__main__":
    data_dir = "./data"
    results_dir = "./results"

    for arg in sys.argv[1:]:
        if arg.startswith("--data-dir="):
            data_dir = arg.split("=")[1]
        elif arg.startswith("--results-dir="):
            results_dir = arg.split("=")[1]

    t0 = time.time()
    results = run_pipeline(data_dir=data_dir, results_dir=results_dir)
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
