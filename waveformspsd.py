import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================================================
# Output directory and channel names
# ======================================================
out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD/saturation_check"
os.makedirs(out_dir, exist_ok=True)

basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'
psd = "DRS_Board7_Group1_Channel1"

# ======================================================
# File dictionaries
# ======================================================
positron_files = {
    'run1410_250925145231.root': '100', 'run1411_250925154340.root': '120',
    'run1422_250926102502.root': '110', 'run1409_250925135843.root': '80',
    'run1412_250925174602.root': '60', 'run1415_250925192957.root': '40',
    'run1416_250925230347.root': '30', 'run1423_250926105310.root': '20',
    'run1424_250926124313.root': '10',
}
pion_files = {
    'run1433_250926213442.root': '120', 'run1432_250926203416.root': '100',
    'run1434_250926222520.root': '160', 'run1429_250926183919.root': '80',
    'run1437_250927003120.root': '60', 'run1438_250927012632.root': '40',
    'run1439_250927023319.root': '30', 'run1441_250927033539.root': '20',
    'run1442_250927050848.root': '10', 'run1452_250927102123.root': '5',
}
muon_files = {
    'run1447_250927084726.root': '170', 'run1445_250927074156.root': '110'
}

# ======================================================
# Function to plot top-N largest-peak PSD waveforms
# ======================================================
def plot_largest_peaks(file_path, psd, particle, energy, n_peaks=10):
    run_number = file_path.split('/')[-1].split('_')[0].replace("run", "")
    print(f"Processing {particle} {energy} GeV (Run {run_number})")

    # --- Load waveforms ---
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        waveforms = tree[psd].array(library="np")

    # --- Find peak amplitude (minimum value since signals are negative) ---
    min_vals = np.min(waveforms, axis=1)  # each event’s min sample
    # take indices of the most negative values (deepest dips)
    top_indices = np.argsort(min_vals)[:n_peaks]

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    for idx in top_indices:
        wf = waveforms[idx]
        plt.plot(wf, alpha=0.8, label=f"Event {idx}, min={min_vals[idx]:.0f}")
    plt.xlabel("Sample index")
    plt.ylabel("ADC counts")
    plt.title(f"PSD Waveforms with Largest Peaks\n{particle} {energy} GeV (Run {run_number})")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Save ---
    pdf_path = os.path.join(out_dir, f"{particle}_run{run_number}_{energy}GeV_topPeaks.pdf")
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {pdf_path}")

# ======================================================
# Run for all particle types
# ======================================================
def process_all(files_dict, particle):
    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        plot_largest_peaks(file_path, psd, particle, energy)

process_all(positron_files, "positrons")
process_all(pion_files, "pions")
process_all(muon_files, "muons")
