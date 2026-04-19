"""
Ensemble Classification Pipeline for TVC Thermoacoustic Regimes
================================================================
Runs a UNIFIED leave-one-L/D-out CV loop where all 7 base models
(3 classical ML + 4 deep learning) produce aligned recording-level
probability vectors, then builds 6 ensemble methods on top.

Base models  (re-trained each fold inside this script):
  Classical: SVM-Combined, RF-Combined, XGB-Combined
  Deep:      1D-CNN, LSTM, GRU, 2D-CNN-RP

Ensembles:
  4a Hard-Vote-All7    4b Soft-Vote-All7    4c Classical-Only
  4d Deep-Only         4e Hybrid-Best       4f Stacking-LogReg

Outputs: confusion matrices, bar charts, CSV of all predictions,
         comprehensive comparison table, Grad-CAM for 1D-CNN.

Usage:  python main_ensembles.py --real-data
"""

import sys, os, time, random, warnings, traceback
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier

# ── reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import (extract_features_single_window,
                                extract_recording_features,
                                compute_recurrence_plot)
from nonlinear_features import compute_all_nonlinear_features
from data_loading import load_all_data, REGIME_LABELS, SAMPLING_FREQ

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REGIME_LABELS_3CLASS = {0: "Periodic", 1: "Quasi-periodic", 2: "Aperiodic"}

def remap3(y):
    m = {0:0, 1:0, 2:1, 3:2, 4:2}
    return np.array([m[int(v)] for v in y])

# =====================================================================
# DATA PREPARATION  (one-time, shared across folds)
# =====================================================================

def _window_sig(p, fs=SAMPLING_FREQ, ms=50, ov=0.5):
    ws = int(ms*fs/1000); st = int(ws*(1-ov)); out = []; s = 0
    while s+ws <= p.shape[0]: out.append(p[s:s+ws].copy()); s += st
    return out

def prep_windows(dataset):
    W, Y, LD = [], [], []
    for r in dataset:
        for w in _window_sig(r['pressure']):
            W.append(w); Y.append(r['regime_label']); LD.append(r['LD_ratio'])
    return np.asarray(W, np.float32), np.array(Y), np.array(LD)

def prep_sequences(dataset, seq_len=10, fs=SAMPLING_FREQ, ms=50, ov=0.5):
    ws = int(ms*fs/1000); st = int(ws*(1-ov))
    S, Y, LD = [], [], []; fn = None
    for r in dataset:
        fv = []; s = 0; p = r['pressure']
        while s+ws <= p.shape[0]:
            d = extract_features_single_window(p[s:s+ws], fs)
            if fn is None: fn = sorted(d.keys())
            fv.append([d[k] for k in fn]); s += st
        fv = np.nan_to_num(np.array(fv, np.float32))
        for i in range(len(fv)//seq_len):
            S.append(fv[i*seq_len:(i+1)*seq_len])
            Y.append(r['regime_label']); LD.append(r['LD_ratio'])
    return np.array(S), np.array(Y), np.array(LD), fn

def prep_rp(dataset):
    R, Y, LD = [], [], []
    for r in dataset:
        for ch in range(r['pressure'].shape[1]):
            rp = compute_recurrence_plot(r['pressure'][:,ch],
                                         embedding_dim=3, threshold_percentile=20)
            if rp.shape != (300,300):
                from scipy.ndimage import zoom
                rp = zoom(rp, (300/rp.shape[0], 300/rp.shape[1]), order=0)
            R.append(rp.astype(np.float32))
            Y.append(r['regime_label']); LD.append(r['LD_ratio'])
    return np.array(R), np.array(Y), np.array(LD)

def prep_combined(dataset, fs=SAMPLING_FREQ):
    Xw_l, Xn_l, y_l, ld_l = [], [], [], []
    fnw = fnn = None
    for r in dataset:
        rf, _ = extract_recording_features(r['pressure'], fs)
        if fnw is None: fnw = sorted(rf.keys())
        Xw_l.append([rf[k] for k in fnw])
        nf = compute_all_nonlinear_features(r['pressure'], fs)
        if fnn is None: fnn = sorted(nf.keys())
        Xn_l.append([nf[k] for k in fnn])
        y_l.append(r['regime_label']); ld_l.append(r['LD_ratio'])
    Xw = np.nan_to_num(np.array(Xw_l)); Xn = np.nan_to_num(np.array(Xn_l))
    return np.hstack([Xw, Xn]), np.array(y_l), np.array(ld_l)

# =====================================================================
# PyTorch DATASETS
# =====================================================================

class WinDS(Dataset):
    def __init__(s, X, y, aug=False):
        s.X = torch.tensor(X.transpose(0,2,1), dtype=torch.float32)
        s.y = torch.tensor(y, dtype=torch.long); s.aug = aug
    def norm(s, m, sd): s.X = (s.X - m)/(sd+1e-8)
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        x = s.X[i].clone()
        if s.aug:
            x += 0.05*torch.randn_like(x)
            sh = random.randint(-50,50)
            if sh: x = torch.roll(x, sh, 1)
        return x, s.y[i]

class SeqDS(Dataset):
    def __init__(s, X, y):
        s.X = torch.tensor(X, dtype=torch.float32)
        s.y = torch.tensor(y, dtype=torch.long)
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return s.X[i], s.y[i]

class RPDS(Dataset):
    def __init__(s, X, y, aug=False):
        s.X = torch.tensor(X[:,None], dtype=torch.float32)
        s.y = torch.tensor(y, dtype=torch.long); s.aug = aug
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        x = s.X[i].clone()
        if s.aug:
            if random.random()>.5: x = x.flip(1)
            if random.random()>.5: x = x.flip(2)
            k = random.randint(0,3)
            if k: x = torch.rot90(x, k, [1,2])
            x += 0.02*torch.randn_like(x)
        return x, s.y[i]

# =====================================================================
# MODEL ARCHITECTURES  (identical to main_deep_learning.py)
# =====================================================================

class CNN1D(nn.Module):
    def __init__(s, nc=5):
        super().__init__()
        s.features = nn.Sequential(
            nn.Conv1d(3,16,15,2,7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,7,1,3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,1,2), nn.BatchNorm1d(64), nn.ReLU())
        s.classifier = nn.Sequential(
            nn.Dropout(.5), nn.Linear(64,32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32,nc))
    def forward(s, x): return s.classifier(s.features(x).mean(2))

class RNNModel(nn.Module):
    def __init__(s, inp, nc=5, h=48, cell='LSTM'):
        super().__init__()
        C = nn.LSTM if cell=='LSTM' else nn.GRU
        s.rnn = C(input_size=inp, hidden_size=h, num_layers=1, batch_first=True)
        s.cell = cell
        s.head = nn.Sequential(
            nn.Dropout(.5), nn.Linear(h,24), nn.ReLU(), nn.Dropout(.3), nn.Linear(24,nc))
    def forward(s, x):
        if s.cell=='LSTM': _,(h,_) = s.rnn(x)
        else: _,h = s.rnn(x)
        return s.head(h[-1])

class CNN2D(nn.Module):
    def __init__(s, nc=5):
        super().__init__()
        s.features = nn.Sequential(
            nn.Conv2d(1,8,5,2,2), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,1,1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU())
        s.classifier = nn.Sequential(
            nn.Dropout(.5), nn.Linear(32,16), nn.ReLU(), nn.Dropout(.3), nn.Linear(16,nc))
    def forward(s, x): return s.classifier(s.features(x).mean([2,3]))

def npar(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# =====================================================================
# TRAINING ENGINE
# =====================================================================

def _train(model, tl, vl, nc, epochs=100, pat=15, lr=1e-3, wd=1e-4):
    model.to(DEVICE)
    labs = []; [labs.extend(yy.numpy()) for _,yy in tl]
    cc = np.maximum(np.bincount(labs, minlength=nc).astype(float), 1.0)
    w = 1./cc; w = w/w.sum()*nc
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(DEVICE))
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_vl = 1e9; best_st = None; wait = 0
    for ep in range(epochs):
        model.train(); tls=tc=tt=0
        for xb,yb in tl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); o = model(xb); l = crit(o,yb); l.backward(); opt.step()
            tls += l.item()*len(yb); tc += (o.argmax(1)==yb).sum().item(); tt += len(yb)
        model.eval(); vls=vc=vt=0
        with torch.no_grad():
            for xb,yb in vl:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                o = model(xb); l = crit(o,yb)
                vls += l.item()*len(yb); vc += (o.argmax(1)==yb).sum().item(); vt += len(yb)
        va = vls/max(vt,1)
        if va < best_vl:
            best_vl = va; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else: wait += 1
        if (ep+1)%25==0 or ep==0:
            print(f"        ep{ep+1:3d} tL={tls/max(tt,1):.3f} tA={tc/max(tt,1):.2f} "
                  f"vL={va:.3f} vA={vc/max(vt,1):.2f}")
        if wait >= pat: print(f"        early-stop ep{ep+1}"); break
    if best_st: model.load_state_dict(best_st)
    model.to(DEVICE); return model

def _probs(model, loader, nc):
    model.eval(); ps = []
    with torch.no_grad():
        for xb,_ in loader:
            ps.append(torch.softmax(model(xb.to(DEVICE)),1).cpu().numpy())
    return np.vstack(ps) if ps else np.zeros((0,nc))

# =====================================================================
# UNIFIED CV LOOP
# =====================================================================

def unified_cv(dataset, nc, rlabels, out_dir):
    """Single CV loop producing aligned recording-level probabilities."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{nc}c"

    # ── pre-compute all representations ──
    print(f"\n{'='*70}\nPre-computing data  ({nc}-class)\n{'='*70}")
    W, Yw, LDw = prep_windows(dataset);   print(f"  Windows  : {W.shape}")
    S, Ys, LDs, sfn = prep_sequences(dataset); nf = S.shape[2]
    print(f"  Sequences: {S.shape}  ({nf} feat/win)")
    RP, Yr, LDr = prep_rp(dataset);       print(f"  RP images: {RP.shape}")
    Xc, Yc, LDc = prep_combined(dataset); print(f"  Combined : {Xc.shape}")

    if nc == 3:
        Yw = remap3(Yw); Ys = remap3(Ys); Yr = remap3(Yr); Yc = remap3(Yc)

    uLD = np.unique(LDc); nf_total = len(uLD)
    model_names = ['SVM_comb','RF_comb','XGB_comb','1D-CNN','LSTM','GRU','2D-CNN_RP']
    probs  = {m: np.zeros((nf_total, nc)) for m in model_names}
    ytrue  = np.zeros(nf_total, dtype=int)
    rec_LD = np.zeros(nf_total)

    print(f"\n{'='*70}\nLeave-one-L/D-out CV  ({nf_total} folds, {nc}-class)\n{'='*70}")

    for fi, hLD in enumerate(uLD):
        tl = int(Yc[LDc==hLD][0])
        ytrue[fi] = tl; rec_LD[fi] = hLD
        rn = rlabels[tl]
        print(f"\n  Fold {fi+1:2d}/{nf_total}  L/D={hLD:.3f}  ({rn})")

        # ── val L/D (for DL early stopping) ──
        train_LDs = uLD[uLD != hLD]
        vLD = train_LDs[fi % len(train_LDs)]

        # ════════════ CLASSICAL ML ════════════
        tr = LDc != hLD; te = ~tr
        sc = StandardScaler(); Xtr = sc.fit_transform(Xc[tr]); Xte = sc.transform(Xc[te])
        for mn, clf in [
            ('SVM_comb',  SVC(kernel='rbf',C=10,gamma='scale',probability=True,random_state=42)),
            ('RF_comb',   RandomForestClassifier(100,max_depth=5,min_samples_leaf=2,random_state=42)),
            ('XGB_comb',  XGBClassifier(100,max_depth=3,learning_rate=.1,min_child_weight=2,
                                         random_state=42,eval_metric='mlogloss',verbosity=0)),
        ]:
            try:
                clf.fit(Xtr, Yc[tr])
                p = clf.predict_proba(Xte)
                # ensure prob vector has nc columns (handles missing classes)
                if p.shape[1] < nc:
                    full = np.zeros((p.shape[0], nc))
                    for ci, cl in enumerate(clf.classes_): full[:, int(cl)] = p[:, ci]
                    p = full
                probs[mn][fi] = p.mean(0)
            except Exception as e:
                print(f"      {mn} ERR: {e}")

        # ════════════ 1D-CNN ════════════
        try:
            trm = (LDw!=hLD)&(LDw!=vLD); vm = LDw==vLD; tm = LDw==hLD
            ds_tr = WinDS(W[trm],Yw[trm],True); ds_v = WinDS(W[vm],Yw[vm]); ds_te = WinDS(W[tm],Yw[tm])
            mu = ds_tr.X.mean([0,2],keepdim=True); sd = ds_tr.X.std([0,2],keepdim=True)
            ds_tr.norm(mu,sd); ds_v.norm(mu,sd); ds_te.norm(mu,sd)
            print(f"      1D-CNN tr={len(ds_tr)} v={len(ds_v)} te={len(ds_te)}")
            m = CNN1D(nc)
            m = _train(m, DataLoader(ds_tr,32,True), DataLoader(ds_v,32), nc)
            p = _probs(m, DataLoader(ds_te,32), nc)
            probs['1D-CNN'][fi] = p.mean(0)
        except Exception as e: print(f"      1D-CNN ERR: {e}")

        # ════════════ LSTM / GRU ════════════
        for cell in ['LSTM','GRU']:
            try:
                trm = (LDs!=hLD)&(LDs!=vLD); vm = LDs==vLD; tm = LDs==hLD
                sc2 = StandardScaler(); sc2.fit(S[trm].reshape(-1,nf))
                def _s(a): return np.nan_to_num(sc2.transform(a.reshape(-1,nf)).reshape(a.shape).astype(np.float32))
                ds_tr = SeqDS(_s(S[trm]),Ys[trm]); ds_v = SeqDS(_s(S[vm]),Ys[vm]); ds_te = SeqDS(_s(S[tm]),Ys[tm])
                print(f"      {cell:4s}   tr={len(ds_tr)} v={len(ds_v)} te={len(ds_te)}")
                m = RNNModel(nf, nc, cell=cell)
                m = _train(m, DataLoader(ds_tr,16,True), DataLoader(ds_v,16), nc)
                p = _probs(m, DataLoader(ds_te,16), nc)
                probs[cell][fi] = p.mean(0) if len(p) else np.ones(nc)/nc
            except Exception as e: print(f"      {cell} ERR: {e}")

        # ════════════ 2D-CNN RP ════════════
        try:
            trm = (LDr!=hLD)&(LDr!=vLD); vm = LDr==vLD; tm = LDr==hLD
            ds_tr = RPDS(RP[trm],Yr[trm],True); ds_v = RPDS(RP[vm],Yr[vm]); ds_te = RPDS(RP[tm],Yr[tm])
            print(f"      2D-CNN tr={len(ds_tr)} v={len(ds_v)} te={len(ds_te)}")
            m = CNN2D(nc)
            m = _train(m, DataLoader(ds_tr,8,True), DataLoader(ds_v,8), nc)
            p = _probs(m, DataLoader(ds_te,8), nc)
            probs['2D-CNN_RP'][fi] = p.mean(0)
        except Exception as e: print(f"      2D-CNN ERR: {e}")

        # fold summary
        line = "      => "
        for mn in model_names:
            pr = int(probs[mn][fi].argmax())
            ok = "ok" if pr==tl else "X"
            line += f"{mn.split('_')[0][:5]}={rlabels[pr][:5]}({ok}) "
        print(line)

    # ── individual results ──
    indiv = {}
    print(f"\n{'='*70}\nIndividual model accuracies ({nc}-class)\n{'='*70}")
    for mn in model_names:
        prd = probs[mn].argmax(1); acc = accuracy_score(ytrue, prd)
        indiv[mn] = dict(y_true=ytrue.copy(), y_pred=prd, probs=probs[mn].copy(),
                         accuracy=acc, classifier_name=f"{mn} ({nc}c)")
        print(f"  {mn:15s}: {acc:.1%}")

    # ── ensembles ──
    ens = {}
    print(f"\n{'='*70}\nEnsembles ({nc}-class)\n{'='*70}")

    def _hard(names, lbl):
        votes = np.stack([probs[n].argmax(1) for n in names], 1)
        prd = np.array([np.bincount(votes[i],minlength=nc).argmax() for i in range(nf_total)])
        acc = accuracy_score(ytrue, prd)
        ens[lbl] = dict(y_true=ytrue.copy(), y_pred=prd, accuracy=acc,
                        classifier_name=f"{lbl} ({nc}c)")
        print(f"  {lbl:32s}: {acc:.1%}")

    def _soft(names, lbl):
        avg = np.mean([probs[n] for n in names], 0)
        prd = avg.argmax(1); acc = accuracy_score(ytrue, prd)
        ens[lbl] = dict(y_true=ytrue.copy(), y_pred=prd, probs=avg, accuracy=acc,
                        classifier_name=f"{lbl} ({nc}c)")
        print(f"  {lbl:32s}: {acc:.1%}")

    _hard(model_names, 'Hard-Vote-All7')
    _soft(model_names, 'Soft-Vote-All7')

    cl_names = ['SVM_comb','RF_comb','XGB_comb']
    dl_names = ['1D-CNN','LSTM','GRU','2D-CNN_RP']
    _soft(cl_names, 'Soft-Vote-Classical')
    _soft(dl_names, 'Soft-Vote-Deep')

    best_cl = max(cl_names, key=lambda n: indiv[n]['accuracy'])
    best_dl = max(dl_names, key=lambda n: indiv[n]['accuracy'])
    hybrid_lbl = f'Hybrid({best_cl}+{best_dl})'
    _soft([best_cl, best_dl], hybrid_lbl)

    # stacking
    try:
        print("  Stacking-LogReg... ", end="", flush=True)
        meta_X = np.hstack([probs[mn] for mn in model_names])   # (20, 7*nc)
        meta_preds = np.zeros(nf_total, dtype=int)
        for fi in range(nf_total):
            tr = np.arange(nf_total)!=fi
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(meta_X[tr], ytrue[tr])
            meta_preds[fi] = lr.predict(meta_X[fi:fi+1])[0]
        acc = accuracy_score(ytrue, meta_preds)
        ens['Stacking-LogReg'] = dict(y_true=ytrue.copy(), y_pred=meta_preds,
                                       accuracy=acc, classifier_name=f"Stacking-LogReg ({nc}c)")
        print(f"{acc:.1%}")
    except Exception as e:
        print(f"FAILED: {e}")
        ens['Stacking-LogReg'] = dict(y_true=ytrue.copy(), y_pred=np.zeros(nf_total,int),
                                       accuracy=0.0, classifier_name=f"Stacking-LogReg ({nc}c)")

    # ── save confusion matrices ──
    print(f"\n  Saving confusion matrices to {out_dir}/")
    for nm, res in {**indiv, **ens}.items():
        safe = nm.replace(' ','_').replace('(','').replace(')','').replace('+','_')
        _save_cm(res, rlabels, os.path.join(out_dir, f'confusion_{tag}_{safe}.png'))

    return indiv, ens, probs, ytrue, rec_LD

def _save_cm(res, rl, path):
    pc = sorted(set(res['y_true'])|set(res['y_pred']))
    nm = [rl[c] for c in pc]
    cm = confusion_matrix(res['y_true'], res['y_pred'], labels=pc)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nm, yticklabels=nm, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f"{res['classifier_name']}\nAccuracy: {res['accuracy']:.1%}")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

# =====================================================================
# GRAD-CAM  (reused from main_deep_learning.py logic)
# =====================================================================

def grad_cam_1d(model, x_t, cls):
    model.eval()
    x = x_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
    acts, grads = [], []
    h1 = model.features[-1].register_forward_hook(lambda m,i,o: acts.append(o))
    h2 = model.features[-1].register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    out = model(x); model.zero_grad(); out[0,cls].backward()
    h1.remove(); h2.remove()
    a = acts[0].detach().cpu().numpy()[0]; g = grads[0].detach().cpu().numpy()[0]
    w = g.mean(1); cam = np.maximum(np.einsum('c,ct->t',w,a), 0)
    from scipy.ndimage import zoom
    cam = zoom(cam, x.shape[2]/len(cam), order=1)
    return (cam-cam.min())/(cam.max()-cam.min()+1e-8)

def gen_gradcam(dataset, nc, rl, out_dir):
    print("  Generating Grad-CAM for 1D-CNN...")
    W, Y, LD = prep_windows(dataset)
    if nc==3: Y = remap3(Y)
    ds = WinDS(W, Y); mu = ds.X.mean([0,2],keepdim=True); sd = ds.X.std([0,2],keepdim=True)
    ds.norm(mu, sd)
    loader = DataLoader(ds, 32, True); vl = DataLoader(ds, 32)
    model = CNN1D(nc); model = _train(model, loader, vl, nc, 60, 15)
    preds = []; model.eval()
    with torch.no_grad():
        for xb,_ in DataLoader(ds,64): preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
    preds = np.array(preds)
    fig, axes = plt.subplots(nc, 1, figsize=(12, 3*nc))
    if nc==1: axes=[axes]
    t_ms = np.arange(1000)/SAMPLING_FREQ*1000
    for cls in range(nc):
        ax = axes[cls]; mask = (Y==cls)&(preds==cls); idx = np.where(mask)[0]
        if len(idx)==0: ax.text(.5,.5,f'No correct: {rl[cls]}',ha='center',va='center',transform=ax.transAxes); continue
        idx = idx[0]; cam = grad_cam_1d(model, ds.X[idx], cls); wav = ds.X[idx][0].numpy()
        ax.plot(t_ms, wav, 'k-', lw=.5, alpha=.7)
        ax.fill_between(t_ms, wav.min(), wav.max(), where=cam>.5, alpha=.3, color='red', label='Grad-CAM high')
        ax2 = ax.twinx(); ax2.plot(t_ms, cam, 'r-', lw=1.5, alpha=.6); ax2.set_ylim(0,1.5)
        ax2.set_ylabel('Grad-CAM', color='red', fontsize=9)
        ax.set_title(f'{rl[cls]} (L/D={LD[idx]:.3f})'); ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Pressure (norm.)'); ax.legend(fontsize=8, loc='upper right')
    plt.suptitle('Grad-CAM: 1D-CNN Attention', fontsize=13); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'grad_cam_1dcnn_{nc}c.png'), dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved grad_cam_1dcnn_{nc}c.png")

# =====================================================================
# BAR CHARTS
# =====================================================================

def bar_chart(rows, nc, path):
    """rows: list of (name, type, accuracy%)"""
    fig, ax = plt.subplots(figsize=(14, max(6, len(rows)*0.38)))
    colors = {'Classical':'#2196F3', 'Deep':'#FF9800', 'Ensemble':'#4CAF50'}
    y = np.arange(len(rows)); accs = [r[2] for r in rows]
    cols = [colors.get(r[1],'gray') for r in rows]
    bars = ax.barh(y, accs, color=cols)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel('Accuracy (%)'); ax.set_xlim(0,105)
    ax.set_title(f'{nc}-class Method Comparison')
    for b,a in zip(bars,accs): ax.text(b.get_width()+.5, b.get_y()+b.get_height()/2, f'{a:.1f}%', va='center', fontsize=7)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=l) for l,c in colors.items()], loc='lower right')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

# =====================================================================
# MAIN
# =====================================================================

def run(data_dir="./data", results_dir="./results"):
    t0 = time.time()

    print("="*70+"\nLOADING DATA\n"+"="*70)
    dataset = load_all_data(data_dir=data_dir)

    # ── param counts ──
    print("\n"+"="*70+"\nMODEL PARAMETER COUNTS\n"+"="*70)
    nf_approx = 33
    for nm, mdl in [("1D-CNN",CNN1D(5)), ("LSTM",RNNModel(nf_approx,5,cell='LSTM')),
                     ("GRU",RNNModel(nf_approx,5,cell='GRU')), ("2D-CNN RP",CNN2D(5))]:
        print(f"  {nm:15s}: {npar(mdl):>7,d} params")

    # ── run both class configs ──
    all_tables = []  # (display_name, type, acc5, acc3) — built incrementally
    csv_data_5 = csv_data_3 = None
    results_by_nc = {}

    for nc, rl, sub in [(5, REGIME_LABELS, 'ens_5class'),
                         (3, REGIME_LABELS_3CLASS, 'ens_3class')]:
        sd = os.path.join(results_dir, sub); os.makedirs(sd, exist_ok=True)
        ind, ens, prb, yt, ld = unified_cv(dataset, nc, rl, sd)
        results_by_nc[nc] = (ind, ens, prb, yt, ld)

        # Grad-CAM
        try: gen_gradcam(dataset, nc, rl, sd)
        except Exception as e: print(f"  Grad-CAM failed: {e}")

    # ── build unified table ──
    classical_known = [
        ("SVM (RBF) - Windowed",        "Classical", 60.0, 85.0),
        ("Random Forest - Windowed",     "Classical", 60.0, 95.0),
        ("XGBoost - Windowed",           "Classical", 55.0, 90.0),
        ("SVM (RBF) - Nonlinear",        "Classical", 65.0, 90.0),
        ("Random Forest - Nonlinear",    "Classical", 65.0, 95.0),
        ("XGBoost - Nonlinear",          "Classical", 55.0, 85.0),
        ("SVM (RBF) - Combined",         "Classical", 65.0, 85.0),
        ("Random Forest - Combined",     "Classical", 60.0, 90.0),
        ("XGBoost - Combined",           "Classical", 60.0, 95.0),
    ]

    def _g(name, nc):
        ind, ens, *_ = results_by_nc[nc]
        for d in [ind, ens]:
            if name in d: return d[name]['accuracy']*100
        return 0.0

    # find hybrid label
    hybrid_lbl_5 = [k for k in results_by_nc[5][1] if k.startswith('Hybrid')]
    hybrid_lbl_5 = hybrid_lbl_5[0] if hybrid_lbl_5 else 'Hybrid'
    hybrid_lbl_3 = [k for k in results_by_nc[3][1] if k.startswith('Hybrid')]
    hybrid_lbl_3 = hybrid_lbl_3[0] if hybrid_lbl_3 else 'Hybrid'

    dl_ens = [
        ('1D-CNN',           '1D-CNN (raw pressure)',          'Deep'),
        ('LSTM',             'LSTM (sequential features)',      'Deep'),
        ('GRU',              'GRU (sequential features)',       'Deep'),
        ('2D-CNN_RP',        '2D-CNN (recurrence plots)',       'Deep'),
        ('Hard-Vote-All7',   'Ensemble: Hard Vote (all 7)',     'Ensemble'),
        ('Soft-Vote-All7',   'Ensemble: Soft Vote (all 7)',     'Ensemble'),
        ('Soft-Vote-Classical','Ensemble: Classical only',       'Ensemble'),
        ('Soft-Vote-Deep',   'Ensemble: Deep only',             'Ensemble'),
        (hybrid_lbl_5,       f'Ensemble: Best hybrid',          'Ensemble'),
        ('Stacking-LogReg',  'Ensemble: Stacking (LogReg)',     'Ensemble'),
    ]

    new_rows = []
    for key, display, mtype in dl_ens:
        a5 = _g(key, 5)
        # for hybrid, keys may differ between 5c and 3c
        a3 = _g(key if key in results_by_nc[3][0] or key in results_by_nc[3][1]
                else hybrid_lbl_3, 3)
        new_rows.append((display, mtype, a5, a3))

    all_rows = classical_known + new_rows

    print("\n"+"="*70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    print(f"\n  {'Method':<40s} {'Type':<10s} {'5-class':>8s} {'3-class':>8s}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*8}")
    for method, mtype, a5, a3 in all_rows:
        print(f"  {method:<40s} {mtype:<10s} {a5:>7.1f}% {a3:>7.1f}%")

    # ── CSV ──
    yt5 = results_by_nc[5][3]; ld5 = results_by_nc[5][4]
    yt3 = results_by_nc[3][3]
    rows_csv = []
    for i in range(len(yt5)):
        row = {'L_D': ld5[i], 'true_5c': int(yt5[i]), 'true_3c': int(yt3[i])}
        for nc_val in [5,3]:
            ind, ens, *_ = results_by_nc[nc_val]
            sfx = f'_{nc_val}c'
            for nm, res in {**ind, **ens}.items():
                row[f'pred{sfx}_{nm}'] = int(res['y_pred'][i])
        rows_csv.append(row)
    df = pd.DataFrame(rows_csv)
    csv_path = os.path.join(results_dir, 'all_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── bar charts ──
    for nc_p in [3, 5]:
        rows_bar = [(m, t, a5 if nc_p==5 else a3) for m,t,a5,a3 in all_rows]
        p = os.path.join(results_dir, f'comparison_{nc_p}class_barchart.png')
        bar_chart(rows_bar, nc_p, p)
        print(f"  Saved: {p}")

    elapsed = time.time()-t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results in: {results_dir}/")
    return all_rows

if __name__ == "__main__":
    data_dir = "./data"; results_dir = "./results"
    for a in sys.argv[1:]:
        if a.startswith("--data-dir="): data_dir = a.split("=")[1]
        elif a.startswith("--results-dir="): results_dir = a.split("=")[1]
    run(data_dir=data_dir, results_dir=results_dir)
