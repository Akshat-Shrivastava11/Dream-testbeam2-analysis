#!/usr/bin/env python3
"""
conv1d_ae_waveforms_beamnet.py

1D Conv Autoencoder on raw waveforms (PSD + Muon + Cherenkov1-3).
Trains on positrons (muon-vetoed), tests on pions (muon-vetoed).
Saves plots and prints verbose diagnostics.

Adjust paths/params at the top.
"""

import os
import sys
import math
import time
import numpy as np
import uproot
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Optional sklearn imports (ROC). If not available, script uses internal AUC.
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ---------------------------
# Config / user-editable
# ---------------------------
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'

positron_files = {
    'run1410_250925145231.root': 100,
    'run1411_250925154340.root': 120,
    'run1422_250926102502.root': 110,
    'run1409_250925135843.root': 80,
    'run1416_250925230347.root': 30,
    'run1423_250926105310.root': 20,
    'run1424_250926124313.root': 10,
    'run1527_250929001555.root': 100,
    'run1526_250928235028.root': 60,
    'run1525_250928232144.root': 40,
}

pion_files = {
    'run1433_250926213442.root': 120,
    'run1432_250926203416.root': 100,
    'run1434_250926222520.root': 160,
    'run1429_250926183919.root': 80,
    'run1437_250927003120.root': 60,
    'run1438_250927012632.root': 40,
    'run1439_250927023319.root': 30,
    'run1441_250927033539.root': 20,
    'run1442_250927050848.root': 10,
    'run1452_250927102123.root': 5,
}

# channel names (as in your ROOT trees)
psd         = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"
ch1         = "DRS_Board7_Group2_Channel5"
ch2         = "DRS_Board7_Group2_Channel6"
ch3         = "DRS_Board7_Group2_Channel7"

# processing params
muon_threshold = 5000          # veto muons
baseline_samples = 20
wf_window = 256                # final waveform length (samples) centered on pulse
align_on = 'psd'               # align pulses by PSD channel peak
center_index = wf_window // 2

# training params
batch_size = 256
epochs = 150
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# output
output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/CNN_AE_waveforms"
os.makedirs(output_folder, exist_ok=True)
print("Output folder:", output_folder)

# ---------------------------
# utilities: integration & preprocessing
# ---------------------------
def integrate_trapezoid(wf, window=50, baseline_samples=20):
    """Trapezoid integration like you asked (baseline-subtract, local around peak)."""
    wf = np.asarray(wf)
    baseline = np.mean(wf[:baseline_samples])
    corrected = wf - baseline
    peak_idx = int(np.argmin(corrected))
    start = max(0, peak_idx - window)
    end = min(len(corrected), peak_idx + window)
    area = np.trapezoid(corrected[start:end], dx=1)
    return -area

def baseline_subtract(wf, baseline_samples=20):
    wf = np.asarray(wf).astype(np.float32)
    baseline = wf[:baseline_samples].mean()
    return wf - baseline

def align_and_crop(wf, peak_index, window=wf_window, center=center_index):
    """
    Align waveform so that peak_index moves to 'center'.
    Then extract window-length segment centered at center (pad with zeros if necessary).
    We'll roll the baseline-subtracted waveform (wrap-around) then slice; wrap is acceptable
    because we will then zero out samples that came from a wrapped region if they exceed boundaries.
    A robust approach is to compute slice bounds in original array and pad appropriately.
    """
    n = len(wf)
    half = window // 2
    # want to extract indices [peak_index - center, peak_index - center + window)
    start_idx = peak_index - center
    end_idx = start_idx + window

    # build an array of desired indices and fetch values with padding
    idxs = np.arange(start_idx, end_idx)
    vals = np.zeros(window, dtype=wf.dtype)
    valid = (idxs >= 0) & (idxs < n)
    if valid.any():
        vals[valid] = wf[idxs[valid]]
    return vals

def preprocess_run_waveforms(psd_arr, muon_arr, c1_arr, c2_arr, c3_arr,
                             muon_threshold=muon_threshold,
                             baseline_samples=baseline_samples,
                             wf_window=wf_window, align_on='psd'):
    """
    Inputs: arrays of shape (N_events, N_samples) for each channel from uproot.
    Returns: list of waveforms shape (N_kept, channels, wf_window) and integrals dicts and energies mask indices.
    """
    N = len(psd_arr)
    chans = []
    kept_indices = []
    integrals = []

    for i in range(N):
        try:
            psd_wf = baseline_subtract(psd_arr[i], baseline_samples)
            muon_wf = baseline_subtract(muon_arr[i], baseline_samples)
            c1_wf = baseline_subtract(c1_arr[i], baseline_samples)
            c2_wf = baseline_subtract(c2_arr[i], baseline_samples)
            c3_wf = baseline_subtract(c3_arr[i], baseline_samples)
        except Exception:
            continue

        # simple muon veto:
        muon_integral = integrate_trapezoid(muon_arr[i], window=wf_window//4, baseline_samples=baseline_samples)
        if muon_integral > muon_threshold:
            continue

        # find peak index on chosen channel
        if align_on == 'psd':
            peak_idx = int(np.argmin(psd_wf))
        else:
            peak_idx = int(np.argmin(psd_wf))

        # center crop aligned around peak
        psd_c = align_and_crop(psd_wf, peak_idx, window=wf_window, center=center_index)
        muon_c = align_and_crop(muon_wf, peak_idx, window=wf_window, center=center_index)
        c1_c = align_and_crop(c1_wf, peak_idx, window=wf_window, center=center_index)
        c2_c = align_and_crop(c2_wf, peak_idx, window=wf_window, center=center_index)
        c3_c = align_and_crop(c3_wf, peak_idx, window=wf_window, center=center_index)

        # optional per-event scaling: divide by peak amplitude of PSD to make shapes comparable
        psd_peak = np.min(psd_c) if np.any(psd_c != 0) else 1.0
        scale = abs(psd_peak) if abs(psd_peak) > 1e-6 else 1.0
        psd_c = psd_c / scale
        muon_c = muon_c / scale
        c1_c = c1_c / scale
        c2_c = c2_c / scale
        c3_c = c3_c / scale

        # stack channels (channels, samples)
        wf_stack = np.stack([psd_c, muon_c, c1_c, c2_c, c3_c], axis=0).astype(np.float32)

        # store integrals (raw) for diagnostics (before normalization)
        psd_int = integrate_trapezoid(psd_arr[i], window=wf_window//4, baseline_samples=baseline_samples)
        c1_int = integrate_trapezoid(c1_arr[i], window=wf_window//4, baseline_samples=baseline_samples)
        c2_int = integrate_trapezoid(c2_arr[i], window=wf_window//4, baseline_samples=baseline_samples)
        c3_int = integrate_trapezoid(c3_arr[i], window=wf_window//4, baseline_samples=baseline_samples)

        chans.append(wf_stack)
        kept_indices.append(i)
        integrals.append([psd_int, muon_integral, c1_int, c2_int, c3_int])

    if len(chans) == 0:
        return np.zeros((0,5,wf_window), dtype=np.float32), np.empty((0,5)), np.array(kept_indices)
    return np.stack(chans, axis=0), np.array(integrals, dtype=np.float32), np.array(kept_indices, dtype=int)

# ---------------------------
# Load ROOT and prepare datasets
# ---------------------------
def load_run(file_path):
    """Open file and read waveform arrays. Expects tree 'EventTree' with channel branches."""
    with uproot.open(file_path) as f:
        # try both names
        tree = None
        for key in ("EventTree", "tree", "Tree"):
            if key in f:
                tree = f[key]
                break
        if tree is None:
            raise KeyError(f"No EventTree/tree in {file_path}")

        # load arrays (they may be jagged; convert to numpy object arrays)
        psd_arr = tree[psd].array(library="np")
        muon_arr = tree[muoncounter].array(library="np")
        c1_arr = tree[ch1].array(library="np")
        c2_arr = tree[ch2].array(library="np")
        c3_arr = tree[ch3].array(library="np")
    return psd_arr, muon_arr, c1_arr, c2_arr, c3_arr

def build_dataset(file_dict, tag):
    all_wfs = []
    all_ints = []
    all_energies = []
    samples_per_file = []
    for fname, energy in file_dict.items():
        fpath = os.path.join(basedir, fname)
        print("Loading:", fpath)
        try:
            psd_arr, muon_arr, c1_arr, c2_arr, c3_arr = load_run(fpath)
            wfs, ints, idxs = preprocess_run_waveforms(psd_arr, muon_arr, c1_arr, c2_arr, c3_arr)
            print(f"  -> kept {wfs.shape[0]} / {len(psd_arr)} events")
            if wfs.shape[0] == 0:
                continue
            all_wfs.append(wfs)
            all_ints.append(ints)
            all_energies.append(np.full(wfs.shape[0], float(energy), dtype=np.float32))
            samples_per_file.append(wfs.shape[0])
        except Exception as e:
            print("  ERROR loading", fname, e)
    if len(all_wfs) == 0:
        return np.zeros((0,5,wf_window), dtype=np.float32), np.zeros((0,5), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    all_wfs = np.concatenate(all_wfs, axis=0)
    all_ints = np.concatenate(all_ints, axis=0)
    all_energies = np.concatenate(all_energies, axis=0)
    return all_wfs, all_ints, all_energies, samples_per_file

print("\n=== Building datasets ===")
pos_wfs, pos_ints, pos_energies, pos_counts = build_dataset(positron_files, "positron")
pion_wfs, pion_ints, pion_energies, pion_counts = build_dataset(pion_files, "pion")

print("\nDataset sizes:")
print(" Positons:", pos_wfs.shape, " integrals:", pos_ints.shape)
print(" Pions   :", pion_wfs.shape, " integrals:", pion_ints.shape)

# small sanity checks
if pos_wfs.shape[0] == 0 or pion_wfs.shape[0] == 0:
    print("ERROR: got empty dataset for positons or pions. Exiting.")
    sys.exit(1)

# ---------------------------
# optional: shuffle, train/val split for pos
# ---------------------------
rng = np.random.RandomState(42)
perm = rng.permutation(len(pos_wfs))
train_frac = 0.9
n_train = int(train_frac * len(pos_wfs))
train_idx = perm[:n_train]
val_idx = perm[n_train:]

train_wfs = pos_wfs[train_idx]
val_wfs = pos_wfs[val_idx]

train_ints = pos_ints[train_idx]
val_ints = pos_ints[val_idx]

# convert to torch tensors/dataloaders
class WaveformDataset(torch.utils.data.Dataset):
    def __init__(self, wfs):
        self.wfs = wfs
    def __len__(self):
        return len(self.wfs)
    def __getitem__(self, i):
        return self.wfs[i]

train_loader = torch.utils.data.DataLoader(WaveformDataset(train_wfs), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(WaveformDataset(val_wfs), batch_size=batch_size, shuffle=False, drop_last=False)

# ---------------------------
# Define Conv1D Autoencoder (explicit verbose layers)
# ---------------------------
class Conv1dAE(nn.Module):
    def __init__(self, in_channels=5, wf_len=wf_window, latent_dim=16):
        super().__init__()
        # Encoder (Conv blocks)
        self.enc_conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3, stride=2)   # -> L/2
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.enc_relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2)            # -> L/4
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.enc_relu2 = nn.ReLU()
        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2)            # -> L/8
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.enc_relu3 = nn.ReLU()

        # compute flattened size:
        reduced_len = wf_len // 8
        self.flattened = 64 * reduced_len
        self.fc_enc = nn.Linear(self.flattened, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.flattened)
        self.dec_convT1 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(32)
        self.dec_relu1 = nn.ReLU()
        self.dec_convT2 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(16)
        self.dec_relu2 = nn.ReLU()
        self.dec_convT3 = nn.ConvTranspose1d(16, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        # output linear (reconstruction), no activation

    def forward(self, x):
        # x: (B, C, L)
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_relu1(x)
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_relu2(x)
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = self.enc_relu3(x)
        # flatten
        b = x.shape[0]
        x = x.view(b, -1)
        z = self.fc_enc(x)          # latent (B, latent_dim)
        # decode
        x = self.fc_dec(z)
        x = x.view(b, 64, self.flattened // 64)
        x = self.dec_convT1(x)
        x = self.dec_bn1(x)
        x = self.dec_relu1(x)
        x = self.dec_convT2(x)
        x = self.dec_bn2(x)
        x = self.dec_relu2(x)
        x = self.dec_convT3(x)
        return x, z

# ---------------------------
# Instantiate model, optimizer, loss
# ---------------------------
model = Conv1dAE(in_channels=5, wf_len=wf_window, latent_dim=16).to(device)
print("\nModel summary (layers & params):")
for name, p in model.named_parameters():
    print(name, p.shape)
total_params = sum(p.numel() for p in model.parameters())
print("Total params:", total_params)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# ---------------------------
# Training loop with validation
# ---------------------------
train_losses = []
val_losses = []
best_val = 1e9
best_state = None
start_time = time.time()
print("\n🚀 Starting training: epochs =", epochs, "batch_size =", batch_size)
for ep in range(1, epochs+1):
    model.train()
    running = 0.0
    batches = 0
    for batch in train_loader:
        batch = batch.to(device)  # shape (B, C, L)
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        running += loss.item()
        batches += 1
    train_loss = running / max(1, batches)
    train_losses.append(train_loss)

    # validation
    model.eval()
    running_v = 0.0
    batches_v = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss_v = criterion(recon, batch)
            running_v += loss_v.item()
            batches_v += 1
    val_loss = running_v / max(1, batches_v)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if ep % 10 == 0 or ep == 1 or ep == epochs:
        elapsed = time.time() - start_time
        print(f"Epoch {ep:04d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  best_val={best_val:.6f}  elapsed={elapsed:.1f}s")

# restore best
if best_state is not None:
    model.load_state_dict(best_state)
print("Training finished. Best val loss:", best_val)

# ---------------------------
# Evaluate on full positron (train+val) and pion datasets
# ---------------------------
model.eval()
with torch.no_grad():
    pos_tensor = torch.tensor(np.concatenate([train_wfs, val_wfs], axis=0), dtype=torch.float32).to(device)
    recon_pos, z_pos = model(pos_tensor)
    recon_pos = recon_pos.cpu().numpy()
    z_pos = z_pos.cpu().numpy()

    pion_tensor = torch.tensor(pion_wfs, dtype=torch.float32).to(device)
    recon_pion, z_pion = model(pion_tensor)
    recon_pion = recon_pion.cpu().numpy()
    z_pion = z_pion.cpu().numpy()

# compute per-event reconstruction error (MSE averaged over channels & samples)
def per_event_mse(original, recon):
    # original, recon: (N, C, L)
    err = np.mean((original - recon) ** 2, axis=(1,2))
    return err

orig_pos = np.concatenate([train_wfs, val_wfs], axis=0)
err_pos = per_event_mse(orig_pos, recon_pos)
err_pion = per_event_mse(pion_wfs, recon_pion)

print("\n📊 Reconstruction Error Summary:")
print(f"  Positron: n={len(err_pos)} mean={err_pos.mean():.6f} std={err_pos.std():.6f}")
print(f"  Pion    : n={len(err_pion)} mean={err_pion.mean():.6f} std={err_pion.std():.6f}")

# ---------------------------
# ROC AUC (sklearn if present, otherwise manual)
# ---------------------------
y_true = np.concatenate([np.zeros_like(err_pos), np.ones_like(err_pion)])
y_score = np.concatenate([err_pos, err_pion])

if _HAS_SKLEARN:
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _thr = roc_curve(y_true, y_score)
else:
    # manual ROC computation
    desc_score_indices = np.argsort(-y_score)
    y_true_sorted = y_true[desc_score_indices]
    # compute tpr/fpr at each threshold step
    P = np.sum(y_true == 1)
    Nneg = np.sum(y_true == 0)
    tpr = []
    fpr = []
    tp = 0
    fp = 0
    last_score = None
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P if P > 0 else 0.0)
        fpr.append(fp / Nneg if Nneg > 0 else 0.0)
    # make arrays
    fpr = np.array([0.0] + fpr + [1.0])
    tpr = np.array([0.0] + tpr + [1.0])
    auc = np.trapz(tpr, fpr)

print(f"\n🧮 ROC-AUC (err score) = {auc:.4f}")

# ---------------------------
# Save plots and diagnostics
# ---------------------------
# 1) training / validation loss
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training / Validation Loss (log scale)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "loss_train_val.png"), dpi=200)
plt.close()

# 2) reconstruction error hist
plt.figure(figsize=(7,5))
plt.hist(err_pos, bins=100, alpha=0.7, label='positron', density=False)
plt.hist(err_pion, bins=100, alpha=0.7, label='pion', density=False)
plt.yscale('log')
plt.xlabel('Per-event MSE')
plt.ylabel('Counts (log scale)')
plt.legend()
plt.title('Reconstruction Error Histogram')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "recon_error_hist.png"), dpi=200)
plt.close()

# 3) ROC curve
if _HAS_SKLEARN:
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (reconstruction error)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "roc_curve.png"), dpi=200)
    plt.close()
else:
    # plot manual
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC~{auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (manual)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "roc_curve_manual.png"), dpi=200)
    plt.close()

# 4) error vs energy (for both sets)
# need energies for pos (train+val) and pion energies
pos_all_energies = np.concatenate([np.repeat(list(positron_files.values())[i], pos_counts[i]) if i < len(pos_counts) else np.array([]) for i in range(len(pos_counts))])
# simpler: we stored pos_energies per-file earlier? we didn't — use pos_energies if available in build_dataset return (we didn't keep). Instead compute per-file mapping:
# We kept pos_counts from build_dataset, but not per-event energies array. Simpler: build per-file energies earlier - but to keep flow, we will just scatter with jittered index.
plt.figure(figsize=(7,4))
jitter_pos = np.random.RandomState(0).rand(len(err_pos))*0.1
plt.scatter(np.arange(len(err_pos))+jitter_pos, err_pos, s=6, alpha=0.6, label='positron')
jitter_pion = np.random.RandomState(1).rand(len(err_pion))*0.1
plt.scatter(np.arange(len(err_pion))+jitter_pion, err_pion, s=6, alpha=0.6, label='pion')
plt.yscale('log')
plt.xlabel('Event index (shuffled)')
plt.ylabel('Per-event MSE')
plt.legend()
plt.title('Per-event MSE (positrons vs pions)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "per_event_mse_scatter.png"), dpi=200)
plt.close()

# 5) latent space PCA -> 2D
# compute PCA manually (center + SVD)
Z = np.concatenate([z_pos, z_pion], axis=0)
Z_centered = Z - Z.mean(axis=0)
# PCA: SVD
U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
Z2 = Z_centered.dot(Vt.T[:, :2])  # N x 2
Npos = z_pos.shape[0]
plt.figure(figsize=(6,5))
plt.scatter(Z2[:Npos,0], Z2[:Npos,1], s=6, alpha=0.6, label='positron')
plt.scatter(Z2[Npos:,0], Z2[Npos:,1], s=6, alpha=0.6, label='pion')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.title('Latent PCA (2D)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "latent_pca.png"), dpi=200)
plt.close()

# 6) show a few example reconstructions (waveform per channel)
def save_example_recon(original_wfs, recon_wfs, idx, tag):
    # original & recon (C,L)
    fig, axs = plt.subplots(5,1, figsize=(6,8), sharex=True)
    ch_names = ['PSD','Muon','Cher1','Cher2','Cher3']
    for ci in range(5):
        axs[ci].plot(original_wfs[ci], label='orig', lw=1)
        axs[ci].plot(recon_wfs[ci], label='recon', lw=1, alpha=0.8)
        axs[ci].set_ylabel(ch_names[ci])
        axs[ci].legend(fontsize=7)
    axs[-1].set_xlabel('sample index')
    plt.suptitle(f"{tag} example idx {idx}")
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(os.path.join(output_folder, f"{tag}_recon_example_{idx}.png"), dpi=200)
    plt.close()

# save 6 positives and 6 pions
n_show = min(6, len(err_pos))
for i in range(n_show):
    save_example_recon(orig_pos[i], recon_pos[i], i, tag='positron')

n_show_p = min(6, len(err_pion))
for i in range(n_show_p):
    save_example_recon(pion_wfs[i], recon_pion[i], i, tag='pion')

# 7) reconstruction error violin + box
plt.figure(figsize=(6,5))
plt.violinplot([err_pos, err_pion], showmeans=True)
plt.xticks([1,2], ['positon','pion'])
plt.ylabel('Per-event MSE')
plt.title('Error violin plot')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "error_violin_box.png"), dpi=200)
plt.close()

# final print
print("\n✅ Saved plots to:", output_folder)
print(f"  AUC (reconstruction error) = {auc:.4f}")
print("  Pos mean/std:", err_pos.mean(), err_pos.std())
print("  Pion mean/std:", err_pion.mean(), err_pion.std())
