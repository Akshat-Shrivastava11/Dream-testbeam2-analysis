# ##THI is beam net
# import os
# import numpy as np
# import uproot
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve

# # ===================================
# # Channels and thresholds
# # ===================================
# psd         = "DRS_Board7_Group1_Channel1"
# muoncounter = "DRS_Board7_Group2_Channel4"

# muon_threshold = 5000
# psd_threshold  = 3000
# window         = 50
# baseline_samples = 20

# basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'
# output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/AutoEncoder_PSDMuon"
# os.makedirs(output_folder, exist_ok=True)

# # ===================================
# # Waveform integration
# # ===================================
# def integrate_waveforms(events, window=50, baseline_samples=20):
#     integrals = []
#     for event in events:
#         event_np = np.array(event)
#         baseline = np.mean(event_np[:baseline_samples])
#         corrected = event_np - baseline
#         peak_index = np.argmin(corrected)
#         start = max(0, peak_index - window)
#         end   = min(len(corrected), peak_index + window)
#         area = np.trapezoid(corrected[start:end], dx=1)
#         integrals.append(-area)
#     return np.array(integrals)

# # ===================================
# # Beam files
# # ===================================
# positron_files = {
#     'run1410_250925145231.root': '100',
#     'run1411_250925154340.root': '120',
#     'run1422_250926102502.root': '110',
#     'run1409_250925135843.root': '80',
#     'run1416_250925230347.root': '30',
#     'run1423_250926105310.root': '20',
#     'run1424_250926124313.root': '10',
#     'run1527_250929001555.root': '100',
#     'run1526_250928235028.root': '60',
#     'run1525_250928232144.root': '40',
# }

# pion_files = {
#     'run1433_250926213442.root': '120',
#     'run1432_250926203416.root': '100',
#     'run1434_250926222520.root': '160',
#     'run1429_250926183919.root': '80',
#     'run1437_250927003120.root': '60',
#     'run1438_250927012632.root': '40',
#     'run1439_250927023319.root': '30',
#     'run1441_250927033539.root': '20',
#     'run1442_250927050848.root': '10',
#     'run1452_250927102123.root': '5',
# }

# muon_files = {
#     'run1447_250927084726.root': '170',
#     'run1445_250927074156.root': '110',
# }

# all_files = {"pion": pion_files, "positron": positron_files, "muon": muon_files}
# data_dict = {"pion": [], "positron": [], "muon": []}

# # ===================================
# # Extract PSD & Muon integrals
# # ===================================
# for particle_type, files_dict in all_files.items():
#     print(f"\n=== Processing {particle_type.upper()} runs ===")
#     for fname, energy in files_dict.items():
#         file_path = os.path.join(basedir, fname)
#         print(f"→ {fname} ({energy} GeV)")

#         try:
#             with uproot.open(file_path) as f:
#                 tree = f["EventTree"]

#                 psd_int  = integrate_waveforms(tree[psd].array(library="np"), window, baseline_samples)
#                 muon_int = integrate_waveforms(tree[muoncounter].array(library="np"), window, baseline_samples)

#             # Masks
#             muon_mask     = muon_int > muon_threshold
#             positron_mask = (muon_int < muon_threshold) & (psd_int > 40000)
#             pion_mask     = (muon_int < muon_threshold) & (psd_int < psd_threshold)

#             # Append data
#             if particle_type == "muon":
#                 data_dict["muon"].append(np.stack([psd_int[muon_mask], muon_int[muon_mask]], axis=1))
#             elif particle_type == "positron":
#                 data_dict["positron"].append(np.stack([psd_int[positron_mask], muon_int[positron_mask]], axis=1))
#             elif particle_type == "pion":
#                 data_dict["pion"].append(np.stack([psd_int[pion_mask], muon_int[pion_mask]], axis=1))

#         except Exception as e:
#             print(f"❌ Failed to process {fname}: {e}")

# normals = np.concatenate(data_dict["muon"] + data_dict["positron"], axis=0)
# pions   = np.concatenate(data_dict["pion"], axis=0)

# print("\n✅ Data summary:")
# print(f"  Muon+Positron samples (normal): {normals.shape[0]}")
# print(f"  Pion samples (anomaly): {pions.shape[0]}")

# # ===================================
# # Plot raw distributions
# # ===================================
# plt.figure(figsize=(6,5))
# plt.scatter(normals[:,0], normals[:,1], s=3, label='Muons+Positrons', alpha=0.5)
# plt.scatter(pions[:,0], pions[:,1], s=3, label='Pions', alpha=0.5)
# plt.xlabel("PSD Integral")
# plt.ylabel("Muon Counter Integral")
# plt.legend()
# plt.title("Raw PSD vs Muon integrals")
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "raw_psd_muon_scatter.pdf"))
# plt.close()

# # ===================================
# # Normalize
# # ===================================
# mean = normals.mean(axis=0)
# std  = normals.std(axis=0)
# normals_std = (normals - mean) / std
# pions_std   = (pions - mean) / std

# # Plot normalized distributions
# plt.figure(figsize=(6,5))
# plt.scatter(normals_std[:,0], normals_std[:,1], s=3, label='Muons+Positrons', alpha=0.5)
# plt.scatter(pions_std[:,0], pions_std[:,1], s=3, label='Pions', alpha=0.5)
# plt.xlabel("Normalized PSD")
# plt.ylabel("Normalized Muon Counter")
# plt.legend()
# plt.title("Normalized PSD vs Muon integrals")
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "normalized_psd_muon_scatter.pdf"))
# plt.close()

# X_train = torch.tensor(normals_std, dtype=torch.float32)
# X_test  = torch.tensor(pions_std, dtype=torch.float32)

# # ===================================
# # Define Autoencoder
# # ===================================
# class SimpleAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(2, 4),
#             nn.ReLU(),
#             nn.Linear(4, 2),
#             nn.ReLU(),
#             nn.Linear(2, 1)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(1, 2),
#             nn.ReLU(),
#             nn.Linear(2, 4),
#             nn.ReLU(),
#             nn.Linear(4, 2)
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

# model = SimpleAE()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # ===================================
# # Training loop
# # ===================================
# epochs = 1000
# losses = []

# print("\n🚀 Training Autoencoder...\n")
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, X_train)
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())

#     if epoch % 5 == 0 or epoch == epochs-1:
#         print(f"Epoch {epoch:03d}/{epochs}: Loss = {loss.item():.6f}")

# # Plot training loss
# plt.figure(figsize=(6,4))
# plt.plot(losses, lw=2)
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.title("Autoencoder Training Loss")
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "training_loss.pdf"))
# plt.close()

# # ===================================
# # Evaluate reconstruction errors
# # ===================================
# model.eval()
# with torch.no_grad():
#     recon_normals = model(X_train)
#     recon_pions   = model(X_test)
#     err_normals = ((recon_normals - X_train)**2).mean(dim=1).numpy()
#     err_pions   = ((recon_pions   - X_test)**2).mean(dim=1).numpy()

# # Print summary stats
# print("\n📊 Reconstruction Error Summary:")
# print(f"  Normal Mean Error: {err_normals.mean():.6f} ± {err_normals.std():.6f}")
# print(f"  Pion Mean Error:   {err_pions.mean():.6f} ± {err_pions.std():.6f}")

# # ===================================
# # Plot error histograms
# # ===================================
# plt.figure(figsize=(7,5))
# plt.hist(err_normals, bins=50, alpha=0.7, label='Muons+Positrons', color='green')
# plt.hist(err_pions, bins=50, alpha=0.7, label='Pions', color='red')
# plt.yscale('log')
# plt.xlabel('Reconstruction Error (MSE)')
# plt.ylabel('Counts (log scale)')
# plt.legend()
# plt.title("Reconstruction Error Distributions")
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "reconstruction_error_hist.pdf"))
# plt.close()

# # ===================================
# # Violin plot for comparison
# # ===================================
# plt.figure(figsize=(6,5))
# plt.violinplot([err_normals, err_pions], showmeans=True)
# plt.xticks([1, 2], ['Muons+Positrons', 'Pions'])
# plt.ylabel('Reconstruction Error (MSE)')
# plt.title("Error Comparison Violin Plot")
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "error_violin.pdf"))
# plt.close()

# # ===================================
# # ROC-style visualization
# # ===================================
# y_true = np.concatenate([np.zeros_like(err_normals), np.ones_like(err_pions)])
# y_score = np.concatenate([err_normals, err_pions])
# auc = roc_auc_score(y_true, y_score)
# fpr, tpr, _ = roc_curve(y_true, y_score)

# print(f"\n🧮 ROC-AUC between normal & pion errors: {auc:.3f}")

# plt.figure(figsize=(6,5))
# plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
# plt.plot([0,1],[0,1],'k--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Anomaly Detection ROC Curve")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_folder, "roc_curve.pdf"))
# plt.close()

# print("\n✅ All plots saved to:", output_folder)



# ===================================================
# BeamNet: Positron vs Pion separation using PSD, Muon, and Cherenkov counters
# ===================================================

import os
import numpy as np
import uproot
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ===================================================
# Channels and thresholds
# ===================================================
psd         = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"
chrnkov1    = "DRS_Board7_Group2_Channel5"
chrnkov2    = "DRS_Board7_Group2_Channel6"
chrnkov3    = "DRS_Board7_Group2_Channel7"

muon_threshold = 5000
psd_threshold  = 3000
window = 50
baseline_samples = 20

basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'
output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/AutoEncoder_PositronPion_Cherekov"
os.makedirs(output_folder, exist_ok=True)

# ===================================================
# Waveform integration
# ===================================================
def integrate_waveforms(events, window=50, baseline_samples=20):
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)
        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)
        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)
    return np.array(integrals)

# ===================================================
# Beam files (POSITRONS and PIONS only)
# ===================================================
positron_files = {
    'run1410_250925145231.root': '100',
    'run1411_250925154340.root': '120',
    'run1422_250926102502.root': '110',
    'run1409_250925135843.root': '80',
    'run1416_250925230347.root': '30',
    'run1423_250926105310.root': '20',
    'run1424_250926124313.root': '10',
    'run1527_250929001555.root': '100',
    'run1526_250928235028.root': '60',
    'run1525_250928232144.root': '40',
}

pion_files = {
    'run1433_250926213442.root': '120',
    'run1432_250926203416.root': '100',
    'run1434_250926222520.root': '160',
    'run1429_250926183919.root': '80',
    'run1437_250927003120.root': '60',
    'run1438_250927012632.root': '40',
    'run1439_250927023319.root': '30',
    'run1441_250927033539.root': '20',
    'run1442_250927050848.root': '10',
    'run1452_250927102123.root': '5',
}

data_dict = {"positron": [], "pion": []}

# ===================================================
# Extract PSD + Muon + Cherenkov integrals
# ===================================================
for particle_type, files_dict in {"positron": positron_files, "pion": pion_files}.items():
    print(f"\n=== Processing {particle_type.upper()} runs ===")
    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        print(f"→ {fname} ({energy} GeV)")

        try:
            with uproot.open(file_path) as f:
                tree = f["EventTree"]

                psd_int  = integrate_waveforms(tree[psd].array(library="np"), window, baseline_samples)
                muon_int = integrate_waveforms(tree[muoncounter].array(library="np"), window, baseline_samples)
                ch1_int  = integrate_waveforms(tree[chrnkov1].array(library="np"), window, baseline_samples)
                ch2_int  = integrate_waveforms(tree[chrnkov2].array(library="np"), window, baseline_samples)
                ch3_int  = integrate_waveforms(tree[chrnkov3].array(library="np"), window, baseline_samples)

            # basic filtering: remove muons
            muon_mask = muon_int < muon_threshold
            psd_f = psd_int[muon_mask]
            muon_f = muon_int[muon_mask]
            ch1_f = ch1_int[muon_mask]
            ch2_f = ch2_int[muon_mask]
            ch3_f = ch3_int[muon_mask]

            data_point = np.stack([psd_f, muon_f, ch1_f, ch2_f, ch3_f], axis=1)
            data_dict[particle_type].append(data_point)

        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

# Concatenate
positrons = np.concatenate(data_dict["positron"], axis=0)
pions     = np.concatenate(data_dict["pion"], axis=0)

print("\n✅ Data summary:")
print(f"  Positron samples: {positrons.shape[0]}")
print(f"  Pion samples:     {pions.shape[0]}")

# ===================================================
# Normalization
# ===================================================
mean = positrons.mean(axis=0)
std  = positrons.std(axis=0)
positrons_std = (positrons - mean) / std
pions_std     = (pions - mean) / std

# ===================================================
# Convert to tensors
# ===================================================
X_train = torch.tensor(positrons_std, dtype=torch.float32)
X_test  = torch.tensor(pions_std, dtype=torch.float32)

# ===================================================
# Verbose Autoencoder Architecture
# ===================================================
class VerboseAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_fc1 = nn.Linear(5, 16)
        self.encoder_act1 = nn.ReLU()
        self.encoder_fc2 = nn.Linear(16, 8)
        self.encoder_act2 = nn.ReLU()
        self.encoder_fc3 = nn.Linear(8, 4)
        self.encoder_act3 = nn.ReLU()
        self.encoder_fc4 = nn.Linear(4, 2)
        self.encoder_act4 = nn.ReLU()
        self.encoder_latent = nn.Linear(2, 1)

        self.decoder_fc1 = nn.Linear(1, 2)
        self.decoder_act1 = nn.ReLU()
        self.decoder_fc2 = nn.Linear(2, 4)
        self.decoder_act2 = nn.ReLU()
        self.decoder_fc3 = nn.Linear(4, 8)
        self.decoder_act3 = nn.ReLU()
        self.decoder_fc4 = nn.Linear(8, 16)
        self.decoder_act4 = nn.ReLU()
        self.decoder_out = nn.Linear(16, 5)

    def forward(self, x):
        x = self.encoder_fc1(x)
        x = self.encoder_act1(x)
        x = self.encoder_fc2(x)
        x = self.encoder_act2(x)
        x = self.encoder_fc3(x)
        x = self.encoder_act3(x)
        x = self.encoder_fc4(x)
        x = self.encoder_act4(x)
        z = self.encoder_latent(x)
        x = self.decoder_fc1(z)
        x = self.decoder_act1(x)
        x = self.decoder_fc2(x)
        x = self.decoder_act2(x)
        x = self.decoder_fc3(x)
        x = self.decoder_act3(x)
        x = self.decoder_fc4(x)
        x = self.decoder_act4(x)
        out = self.decoder_out(x)
        return out

# ===================================================
# Training
# ===================================================
model = VerboseAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 500
losses = []

print("\n🚀 Training Autoencoder...\n")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d}/{epochs}: Loss = {loss.item():.6f}")

# ===================================================
# Plot training loss
# ===================================================
plt.figure(figsize=(6,4))
plt.plot(losses, lw=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_loss.pdf"))
plt.close()

# ===================================================
# Evaluate
# ===================================================
model.eval()
with torch.no_grad():
    recon_pos = model(X_train)
    recon_pion = model(X_test)
    err_pos = ((recon_pos - X_train)**2).mean(dim=1).numpy()
    err_pion = ((recon_pion - X_test)**2).mean(dim=1).numpy()

print("\n📊 Reconstruction Error Summary:")
print(f"  Positron Mean: {err_pos.mean():.6f} ± {err_pos.std():.6f}")
print(f"  Pion Mean:     {err_pion.mean():.6f} ± {err_pion.std():.6f}")

# ===================================================
# Plot reconstruction error histograms
# ===================================================
plt.figure(figsize=(7,5))
plt.hist(err_pos, bins=50, alpha=0.7, label='Positrons', color='blue')
plt.hist(err_pion, bins=50, alpha=0.7, label='Pions', color='red')
plt.yscale('log')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Counts (log scale)')
plt.legend()
plt.title("Reconstruction Error Distributions")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "reconstruction_error_hist.pdf"))
plt.close()

# ===================================================
# ROC Curve
# ===================================================
y_true = np.concatenate([np.zeros_like(err_pos), np.ones_like(err_pion)])
y_score = np.concatenate([err_pos, err_pion])
auc = roc_auc_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)

print(f"\n🧮 ROC-AUC between positrons & pions: {auc:.3f}")

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Anomaly Detection ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "roc_curve.pdf"))
plt.close()

print("\n✅ All plots saved to:", output_folder)
