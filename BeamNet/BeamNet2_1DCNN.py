#!/usr/bin/env python3
"""
Conv1d AE on PSD + Muon + Cherenkov channels
Saves model, training/validation data, and detailed plots.
"""

import os, time, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import uproot

# Optional sklearn for ROC
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False

# ------------------ CONFIG ------------------
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'
output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/BeamNet/AutoEncoder_PSDMuon_gpu"
os.makedirs(output_folder, exist_ok=True)

positron_files = {
    'run1410_250925145231.root': '100',
    'run1411_250925154340.root': '120',

    'run1409_250925135843.root': '80',
    'run1416_250925230347.root': '30',
    'run1423_250926105310.root': '20',
    'run1424_250926124313.root': '10',

    'run1526_250928235028.root': '60',
    'run1525_250928232144.root': '40',
}

pion_files = {
    'run1433_250926213442.root': '120',
    'run1432_250926203416.root': '100',

    'run1429_250926183919.root': '80',
    'run1437_250927003120.root': '60',
    'run1438_250927012632.root': '40',
    'run1439_250927023319.root': '30',
    'run1441_250927033539.root': '20',
    'run1442_250927050848.root': '10',

}

psd, muoncounter, ch1, ch2, ch3 = (
    "DRS_Board7_Group1_Channel1",
    "DRS_Board7_Group2_Channel4",
    "DRS_Board7_Group2_Channel5",
    "DRS_Board7_Group2_Channel6",
    "DRS_Board7_Group2_Channel7",
)

muon_threshold = 5000
baseline_samples = 20
wf_window = 256
center_index = wf_window // 2
batch_size = 256
epochs = 150
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ UTILITIES ------------------
def integrate_trapezoid(wf, window=50, baseline_samples=20):
    wf = np.asarray(wf)
    baseline = np.mean(wf[:baseline_samples])
    corrected = wf - baseline
    peak_idx = int(np.argmin(corrected))
    start = max(0, peak_idx - window)
    end = min(len(corrected), peak_idx + window)
    area = np.trapz(corrected[start:end], dx=1)
    return -area

def baseline_subtract(wf, baseline_samples=20):
    wf = np.asarray(wf, np.float32)
    return wf - np.mean(wf[:baseline_samples])

def align_and_crop(wf, peak_index, window=wf_window, center=center_index):
    n = len(wf)
    start_idx = peak_index - center
    end_idx = start_idx + window
    idxs = np.arange(start_idx, end_idx)
    vals = np.zeros(window, dtype=wf.dtype)
    valid = (idxs >= 0) & (idxs < n)
    vals[valid] = wf[idxs[valid]]
    return vals

def preprocess_run_waveforms(psd_arr, muon_arr, c1_arr, c2_arr, c3_arr):
    chans, integrals, kept_indices = [], [], []
    for i in range(len(psd_arr)):
        psd_wf = baseline_subtract(psd_arr[i])
        muon_wf = baseline_subtract(muon_arr[i])
        c1_wf = baseline_subtract(c1_arr[i])
        c2_wf = baseline_subtract(c2_arr[i])
        c3_wf = baseline_subtract(c3_arr[i])
        muon_int = integrate_trapezoid(muon_arr[i], window=wf_window//4)
        if muon_int > muon_threshold: continue
        peak_idx = int(np.argmin(psd_wf))
        wf_stack = np.stack([
            align_and_crop(psd_wf, peak_idx),
            align_and_crop(muon_wf, peak_idx),
            align_and_crop(c1_wf, peak_idx),
            align_and_crop(c2_wf, peak_idx),
            align_and_crop(c3_wf, peak_idx)
        ], axis=0)
        chans.append(wf_stack)
        integrals.append([
            integrate_trapezoid(psd_arr[i], window=wf_window//4),
            muon_int,
            integrate_trapezoid(c1_arr[i], window=wf_window//4),
            integrate_trapezoid(c2_arr[i], window=wf_window//4),
            integrate_trapezoid(c3_arr[i], window=wf_window//4)
        ])
        kept_indices.append(i)
    if len(chans) == 0:
        return np.zeros((0,5,wf_window)), np.zeros((0,5)), np.array([])
    return np.stack(chans), np.array(integrals, np.float32), np.array(kept_indices)

def load_run(file_path):
    with uproot.open(file_path) as f:
        tree = None
        for key in ("EventTree", "tree", "Tree"):
            if key in f: tree=f[key]; break
        if tree is None: raise KeyError(f"No EventTree/tree in {file_path}")
        return (
            tree[psd].array(library="np"),
            tree[muoncounter].array(library="np"),
            tree[ch1].array(library="np"),
            tree[ch2].array(library="np"),
            tree[ch3].array(library="np")
        )

def build_datasets(file_dict):
    wfs_all, ints_all, E_all = [], [], []
    for fname, energy in file_dict.items():
        fpath = os.path.join(basedir, fname)
        try:
            psd_arr, muon_arr, c1_arr, c2_arr, c3_arr = load_run(fpath)
            wfs, ints, _ = preprocess_run_waveforms(psd_arr, muon_arr, c1_arr, c2_arr, c3_arr)
            if wfs.shape[0]==0: continue
            wfs_all.append(wfs)
            ints_all.append(ints)
            E_all.append(np.full(wfs.shape[0], float(energy)))
        except Exception as e: print("Error loading", fname, e)
    if len(wfs_all)==0: return np.zeros((0,5,wf_window)), np.zeros((0,5)), np.zeros((0,))
    return np.concatenate(wfs_all, axis=0), np.concatenate(ints_all, axis=0), np.concatenate(E_all, axis=0)

# ------------------ LOAD DATA ------------------
pos_wfs, pos_ints, pos_energies = build_datasets(positron_files)
pion_wfs, pion_ints, pion_energies = build_datasets(pion_files)

# ------------------ TRAIN/VAL SPLIT ------------------
rng = np.random.RandomState(42)
perm = rng.permutation(len(pos_wfs))
n_train = int(0.9*len(pos_wfs))
train_idx, val_idx = perm[:n_train], perm[n_train:]
train_wfs, val_wfs = pos_wfs[train_idx], pos_wfs[val_idx]
train_ints, val_ints = pos_ints[train_idx], pos_ints[val_idx]

# Save datasets
np.save(os.path.join(output_folder,"train_wfs.npy"), train_wfs)
np.save(os.path.join(output_folder,"val_wfs.npy"), val_wfs)
np.save(os.path.join(output_folder,"train_ints.npy"), train_ints)
np.save(os.path.join(output_folder,"val_ints.npy"), val_ints)

# ------------------ MODEL ------------------
class Conv1dAE(nn.Module):
    def __init__(self, in_channels=5, wf_len=wf_window, latent_dim=16):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(in_channels,16,7,padding=3,stride=2)
        self.enc_bn1 = nn.BatchNorm1d(16); self.enc_relu1=nn.ReLU()
        self.enc_conv2 = nn.Conv1d(16,32,5,padding=2,stride=2)
        self.enc_bn2 = nn.BatchNorm1d(32); self.enc_relu2=nn.ReLU()
        self.enc_conv3 = nn.Conv1d(32,64,5,padding=2,stride=2)
        self.enc_bn3 = nn.BatchNorm1d(64); self.enc_relu3=nn.ReLU()
        self.flattened = 64*(wf_len//8)
        self.fc_enc = nn.Linear(self.flattened, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened)
        self.dec_convT1 = nn.ConvTranspose1d(64,32,5,stride=2,padding=2,output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(32); self.dec_relu1=nn.ReLU()
        self.dec_convT2 = nn.ConvTranspose1d(32,16,5,stride=2,padding=2,output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(16); self.dec_relu2=nn.ReLU()
        self.dec_convT3 = nn.ConvTranspose1d(16,in_channels,7,stride=2,padding=3,output_padding=1)

    def forward(self,x):
        x=self.enc_conv1(x); x=self.enc_bn1(x); x=self.enc_relu1(x)
        x=self.enc_conv2(x); x=self.enc_bn2(x); x=self.enc_relu2(x)
        x=self.enc_conv3(x); x=self.enc_bn3(x); x=self.enc_relu3(x)
        b=x.shape[0]; x=x.view(b,-1)
        z=self.fc_enc(x)
        x=self.fc_dec(z)
        x=x.view(b,64,self.flattened//64)
        x=self.dec_convT1(x); x=self.dec_bn1(x); x=self.dec_relu1(x)
        x=self.dec_convT2(x); x=self.dec_bn2(x); x=self.dec_relu2(x)
        x=self.dec_convT3(x)
        return x,z

# ------------------ TRAIN ------------------
model = Conv1dAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

class WaveformDataset(torch.utils.data.Dataset):
    def __init__(self, wfs): self.wfs=wfs
    def __len__(self): return len(self.wfs)
    def __getitem__(self,i): return self.wfs[i]

train_loader = torch.utils.data.DataLoader(WaveformDataset(train_wfs), batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(WaveformDataset(val_wfs), batch_size=batch_size, shuffle=False)

train_losses, val_losses, best_val, best_state = [], [], 1e9, None
start_time = time.time()
for ep in range(1,epochs+1):
    model.train(); running=0.; batches=0
    for batch in train_loader:
        batch = batch.to(device); optimizer.zero_grad()
        recon,_ = model(batch); loss=criterion(recon,batch); loss.backward(); optimizer.step()
        running+=loss.item(); batches+=1
    train_loss=running/max(1,batches); train_losses.append(train_loss)
    model.eval(); running_v=0.; batches_v=0
    with torch.no_grad():
        for batch in val_loader:
            batch=batch.to(device); recon,_ = model(batch); loss_v=criterion(recon,batch)
            running_v+=loss_v.item(); batches_v+=1
    val_loss=running_v/max(1,batches_v); val_losses.append(val_loss)
    if val_loss<best_val: best_val=val_loss; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    if ep%10==0 or ep==1 or ep==epochs:
        elapsed=time.time()-start_time
        print(f"Epoch {ep}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val:.6f} elapsed={elapsed:.1f}s")

if best_state is not None: model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(output_folder,"BeamNet_1DCNN_model.pth"))
print("✅ Model saved.")

# ------------------ EVALUATION ------------------
model.eval()
with torch.no_grad():
    pos_tensor=torch.tensor(np.concatenate([train_wfs,val_wfs],axis=0),dtype=torch.float32).to(device)
    recon_pos, z_pos = model(pos_tensor); recon_pos=recon_pos.cpu().numpy(); z_pos=z_pos.cpu().numpy()
    pion_tensor=torch.tensor(pion_wfs,dtype=torch.float32).to(device)
    recon_pion, z_pion = model(pion_tensor); recon_pion=recon_pion.cpu().numpy(); z_pion=z_pion.cpu().numpy()

err_pos = np.mean((np.concatenate([train_wfs,val_wfs])-recon_pos)**2,axis=(1,2))
err_pion = np.mean((pion_wfs-recon_pion)**2,axis=(1,2))
print(f"📊 Positron mean/std: {err_pos.mean():.6f}/{err_pos.std():.6f}")
print(f"📊 Pion mean/std: {err_pion.mean():.6f}/{err_pion.std():.6f}")

# ------------------ ROC ------------------
y_true = np.concatenate([np.zeros_like(err_pos), np.ones_like(err_pion)])
y_score = np.concatenate([err_pos, err_pion])
if _HAS_SKLEARN:
    auc = roc_auc_score(y_true,y_score); fpr,tpr,_ = roc_curve(y_true,y_score)
else:
    desc_score_indices = np.argsort(-y_score); y_true_sorted=y_true[desc_score_indices]
    P,Nneg=np.sum(y_true==1),np.sum(y_true==0); tp,fp,tpr_list,fpr_list=0,0,[0.0],[0.0]
    for val in y_true_sorted:
        if val==1: tp+=1
        else: fp+=1
        tpr_list.append(tp/P if P>0 else 0.0); fpr_list.append(fp/Nneg if Nneg>0 else 0.0)
    tpr,fpr=np.array(tpr_list),np.array(fpr_list); auc=np.trapz(tpr,fpr)
print(f"🧮 ROC-AUC = {auc:.4f}")

# ------------------ PLOTS ------------------
# 1) Training/Validation loss
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

# 2) Reconstruction error histogram (not filled)
plt.figure(figsize=(7,5))
plt.hist(err_pos, bins=100, alpha=0.7, label='positron', histtype='step')
plt.hist(err_pion, bins=100, alpha=0.7, label='pion', histtype='step')
plt.yscale('log')
plt.xlabel('Per-event MSE')
plt.ylabel('Counts (log scale)')
plt.legend()
plt.title('Reconstruction Error Histogram')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "recon_error_hist.png"), dpi=200)
plt.close()

# 3) ROC curve
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "roc_curve.png"), dpi=200)
plt.close()

# 4) Example waveform reconstructions
def plot_example_recon(original, recon, idx, tag):
    fig, axs = plt.subplots(5,1, figsize=(6,8), sharex=True)
    ch_names = ['PSD','Muon','Cher1','Cher2','Cher3']
    for ci in range(5):
        axs[ci].plot(original[ci], label='original', lw=1)
        axs[ci].plot(recon[ci], label='recon', lw=1, alpha=0.8)
        axs[ci].set_ylabel(ch_names[ci])
        axs[ci].legend(fontsize=7)
    axs[-1].set_xlabel('Sample index')
    plt.suptitle(f"{tag} example idx {idx}")
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(os.path.join(output_folder, f"{tag}_recon_example_{idx}.png"), dpi=200)
    plt.close()

n_show = min(6, len(err_pos))
for i in range(n_show):
    plot_example_recon(np.concatenate([train_wfs,val_wfs],axis=0)[i], recon_pos[i], i, 'positron')

n_show_p = min(6, len(err_pion))
for i in range(n_show_p):
    plot_example_recon(pion_wfs[i], recon_pion[i], i, 'pion')

# 5) High-error pion events
high_err_idx = np.argsort(err_pion)[-5:]  # 5 largest
for idx in high_err_idx:
    plot_example_recon(pion_wfs[idx], recon_pion[idx], idx, 'pion_high_error')

# 6) Latent space PCA
try:
    from sklearn.decomposition import PCA
    z_all = np.concatenate([z_pos, z_pion],axis=0)
    labels = np.concatenate([np.zeros(len(z_pos)), np.ones(len(z_pion))])
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_all)
    plt.figure(figsize=(6,5))
    plt.scatter(z_2d[labels==0,0], z_2d[labels==0,1], s=5, label='positron', alpha=0.5)
    plt.scatter(z_2d[labels==1,0], z_2d[labels==1,1], s=5, label='pion', alpha=0.5)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Latent space PCA')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "latent_pca.png"), dpi=200)
    plt.close()
except Exception as e:
    print("PCA latent plot failed:", e)

print("✅ All plots saved in", output_folder)
