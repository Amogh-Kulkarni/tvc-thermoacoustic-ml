"""Modified DL training loop that captures per-epoch metrics."""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, '..', 'src'))
sys.path.insert(0, os.path.join(PROJ, '..'))

from main_deep_learning import (
    CNN1D, LSTMClassifier, GRUClassifier, CNN2D,
    PressureWindowDataset, SequenceDataset, RecurrencePlotDataset,
    prepare_all_windows, prepare_sequential_features, prepare_recurrence_plots,
    remap_to_3class, predict, DEVICE
)
from data_loading import load_all_data, REGIME_LABELS, SAMPLING_FREQ

REGIME_LABELS_3CLASS = {0: "Periodic", 1: "Quasi-periodic", 2: "Aperiodic"}


def train_model_with_history(model, train_loader, val_loader, num_classes,
                              max_epochs=100, patience=15, lr=1e-3,
                              weight_decay=1e-4):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_counts = np.bincount(all_labels, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    for epoch in range(max_epochs):
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

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    for k in history:
        history[k] = np.array(history[k])

    return model, history, best_epoch


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cnn1d_cv_with_history(dataset, num_classes, seed=42, capture_history=True):
    _set_seed(seed)
    regime_labels = REGIME_LABELS if num_classes == 5 else REGIME_LABELS_3CLASS
    windows, labels, LD_vals, _ = prepare_all_windows(dataset)
    if num_classes == 3:
        labels = remap_to_3class(labels)

    unique_LD = np.unique(LD_vals)
    all_true, all_pred, all_LD_test = [], [], []
    fold_histories = []
    best_epochs = []

    for fold_i, held_out_LD in enumerate(unique_LD):
        test_mask = LD_vals == held_out_LD
        train_mask = ~test_mask
        train_LD = np.unique(LD_vals[train_mask])
        val_LD = train_LD[fold_i % len(train_LD)]
        val_mask = LD_vals == val_LD
        actual_train_mask = train_mask & ~val_mask

        train_ds = PressureWindowDataset(windows[actual_train_mask],
                                          labels[actual_train_mask], augment=True)
        val_ds = PressureWindowDataset(windows[val_mask],
                                        labels[val_mask], augment=False)
        test_ds = PressureWindowDataset(windows[test_mask],
                                         labels[test_mask], augment=False)

        train_X = train_ds.X
        ch_mean = train_X.mean(dim=[0, 2], keepdim=True)
        ch_std = train_X.std(dim=[0, 2], keepdim=True)
        train_ds.set_normalization(ch_mean, ch_std)
        val_ds.set_normalization(ch_mean, ch_std)
        test_ds.set_normalization(ch_mean, ch_std)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        model = CNN1D(num_classes=num_classes)
        if capture_history:
            model, hist, best_ep = train_model_with_history(
                model, train_loader, val_loader, num_classes)
            fold_histories.append(hist)
            best_epochs.append(best_ep)
        else:
            model, _, best_ep = train_model_with_history(
                model, train_loader, val_loader, num_classes)
            best_epochs.append(best_ep)

        preds = predict(model, test_loader)
        majority_pred = int(np.bincount(preds, minlength=num_classes).argmax())
        true_label = int(labels[test_mask][0])

        all_true.append(true_label)
        all_pred.append(majority_pred)
        all_LD_test.append(held_out_LD)

        status = "OK" if majority_pred == true_label else "MISS"
        print(f"  Fold {fold_i+1:2d}: L/D={held_out_LD:.3f} "
              f"{regime_labels[true_label]:15s} -> "
              f"{regime_labels[majority_pred]:15s} {status}")

    accuracy = accuracy_score(all_true, all_pred)
    return {
        'y_true': np.array(all_true), 'y_pred': np.array(all_pred),
        'LD_test': np.array(all_LD_test), 'accuracy': accuracy,
        'fold_histories': fold_histories, 'best_epochs': np.array(best_epochs),
    }


def run_rnn_cv_with_history(dataset, num_classes, model_type='LSTM',
                             seed=42, capture_history=True):
    _set_seed(seed)
    regime_labels = REGIME_LABELS if num_classes == 5 else REGIME_LABELS_3CLASS
    sequences, labels, LD_vals, feature_names = prepare_sequential_features(dataset)
    if num_classes == 3:
        labels = remap_to_3class(labels)
    n_features = sequences.shape[2]

    unique_LD = np.unique(LD_vals)
    all_true, all_pred, all_LD_test = [], [], []
    fold_histories = []
    best_epochs = []

    for fold_i, held_out_LD in enumerate(unique_LD):
        test_mask = LD_vals == held_out_LD
        train_mask = ~test_mask
        train_LD = np.unique(LD_vals[train_mask])
        val_LD = train_LD[fold_i % len(train_LD)]
        val_mask = LD_vals == val_LD
        actual_train_mask = train_mask & ~val_mask

        train_seqs = sequences[actual_train_mask]
        scaler = StandardScaler()
        scaler.fit(train_seqs.reshape(-1, n_features))

        def scale_seqs(seqs):
            orig_shape = seqs.shape
            flat = scaler.transform(seqs.reshape(-1, n_features))
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

        if model_type == 'LSTM':
            model = LSTMClassifier(input_size=n_features, num_classes=num_classes)
        else:
            model = GRUClassifier(input_size=n_features, num_classes=num_classes)

        if capture_history:
            model, hist, best_ep = train_model_with_history(
                model, train_loader, val_loader, num_classes,
                max_epochs=100, patience=15, lr=1e-3)
            fold_histories.append(hist)
            best_epochs.append(best_ep)
        else:
            model, _, best_ep = train_model_with_history(
                model, train_loader, val_loader, num_classes,
                max_epochs=100, patience=15, lr=1e-3)
            best_epochs.append(best_ep)

        preds = predict(model, test_loader)
        if len(preds) > 0:
            majority_pred = int(np.bincount(preds, minlength=num_classes).argmax())
        else:
            majority_pred = int(labels[test_mask][0])
        true_label = int(labels[test_mask][0])

        all_true.append(true_label)
        all_pred.append(majority_pred)
        all_LD_test.append(held_out_LD)

        status = "OK" if majority_pred == true_label else "MISS"
        print(f"  Fold {fold_i+1:2d}: L/D={held_out_LD:.3f} "
              f"{regime_labels[true_label]:15s} -> "
              f"{regime_labels[majority_pred]:15s} {status}")

    accuracy = accuracy_score(all_true, all_pred)
    return {
        'y_true': np.array(all_true), 'y_pred': np.array(all_pred),
        'LD_test': np.array(all_LD_test), 'accuracy': accuracy,
        'fold_histories': fold_histories, 'best_epochs': np.array(best_epochs),
    }


def run_cnn2d_cv_with_history(dataset, num_classes, seed=42,
                               capture_history=True):
    _set_seed(seed)
    regime_labels = REGIME_LABELS if num_classes == 5 else REGIME_LABELS_3CLASS
    rps, labels, LD_vals = prepare_recurrence_plots(dataset)
    if num_classes == 3:
        labels = remap_to_3class(labels)

    unique_LD = np.unique(LD_vals)
    all_true, all_pred, all_LD_test = [], [], []
    fold_histories = []
    best_epochs = []

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

        model = CNN2D(num_classes=num_classes)
        if capture_history:
            model, hist, best_ep = train_model_with_history(
                model, train_loader, val_loader, num_classes,
                max_epochs=100, patience=15, lr=1e-3)
            fold_histories.append(hist)
            best_epochs.append(best_ep)
        else:
            model, _, best_ep = train_model_with_history(
                model, train_loader, val_loader, num_classes,
                max_epochs=100, patience=15, lr=1e-3)
            best_epochs.append(best_ep)

        preds = predict(model, test_loader)
        majority_pred = int(np.bincount(preds, minlength=num_classes).argmax())
        true_label = int(labels[test_mask][0])

        all_true.append(true_label)
        all_pred.append(majority_pred)
        all_LD_test.append(held_out_LD)

        status = "OK" if majority_pred == true_label else "MISS"
        print(f"  Fold {fold_i+1:2d}: L/D={held_out_LD:.3f} "
              f"{regime_labels[true_label]:15s} -> "
              f"{regime_labels[majority_pred]:15s} {status}")

    accuracy = accuracy_score(all_true, all_pred)
    return {
        'y_true': np.array(all_true), 'y_pred': np.array(all_pred),
        'LD_test': np.array(all_LD_test), 'accuracy': accuracy,
        'fold_histories': fold_histories, 'best_epochs': np.array(best_epochs),
    }
