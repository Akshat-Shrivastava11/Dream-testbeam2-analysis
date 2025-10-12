import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Branch names ---
muoncounter = "DRS_Board7_Group2_Channel4"
chrnkov1     = "DRS_Board7_Group2_Channel5"
chrnkov2     = "DRS_Board7_Group2_Channel6"
chrnkov3     = "DRS_Board7_Group2_Channel7"

# --- Directories ---
basedir = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/"
outdir  = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/muon_vs_cherenkov/"
os.makedirs(outdir, exist_ok=True)


# --- Integration function ---
def integrate_waveforms(events, window=100, baseline_samples=20):
    """Compute baseline-subtracted integrals for each waveform."""
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)
        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)
        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)  # flip sign to make positive
    return np.array(integrals)


# --- Plot function ---
def plot_muon_vs_cherenkov(run_number, beam_energy, beam_type, muon_ints, chr1, chr2, chr3):
    """3 subplots: Cherenkov (x) vs Muon counter (y)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ch_list = [chr1, chr2, chr3]
    titles = ["Cherenkov 1", "Cherenkov 2", "Cherenkov 3"]

    for i, (ax, ch, title) in enumerate(zip(axes, ch_list, titles)):
        h = ax.hist2d(ch, muon_ints, bins=150, cmap='viridis', cmin=1)
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label("Event count", fontsize=10)
        ax.set_xlabel(f"{title} Integral")
        ax.set_ylabel("Muon Counter Integral")
        ax.set_title(f"{title} vs Muon", fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 40000)
        ax.set_ylim(0, 40000)

    plt.suptitle(f"Muon Counter vs Cherenkov Integrals\nRun {run_number}, {beam_energy} GeV {beam_type}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = os.path.join(outdir, f"Muon_vs_Cherenkov_run{run_number}_{beam_energy}GeV_{beam_type}.pdf")
    plt.savefig(out_file)
    plt.close()
    print(f"✅ Saved Muon–Cherenkov comparison → {out_file}")


# --- Analysis function ---
def analyze_muon_vs_cherenkov(file_path, run_number, beam_energy, beam_type):
    """Load ROOT file, compute integrals for muon + Cherenkov counters, and plot."""
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        muon_data = tree[muoncounter].array(library="np")
        chr1_data = tree[chrnkov1].array(library="np")
        chr2_data = tree[chrnkov2].array(library="np")
        chr3_data = tree[chrnkov3].array(library="np")

    print(f"🔹 Loaded {len(muon_data)} events from {file_path}")

    muon_ints = integrate_waveforms(muon_data)
    chr1_ints = integrate_waveforms(chr1_data)
    chr2_ints = integrate_waveforms(chr2_data)
    chr3_ints = integrate_waveforms(chr3_data)

    plot_muon_vs_cherenkov(run_number, beam_energy, beam_type, muon_ints, chr1_ints, chr2_ints, chr3_ints)


# --- File dictionaries ---
positron_files = {
    'run1410_250925145231.root': '100',
    'run1411_250925154340.root': '120',
    'run1422_250926102502.root': '110',
    'run1409_250925135843.root': '80',
    'run1412_250925174602.root': '60',
    'run1415_250925192957.root': '40',
    'run1416_250925230347.root': '30',
    'run1423_250926105310.root': '20',
    'run1424_250926124313.root': '10',
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

muon_files = {
    'run1447_250927084726.root': '170',
    'run1445_250927074156.root': '110',
}


# --- Main loop ---
def run_all():
    print("=== Processing positron files ===")
    for fname, energy in positron_files.items():
        run_number = fname.split('_')[0].replace("run", "")
        analyze_muon_vs_cherenkov(os.path.join(basedir, fname), run_number, energy, "positron")

    print("=== Processing pion files ===")
    for fname, energy in pion_files.items():
        run_number = fname.split('_')[0].replace("run", "")
        analyze_muon_vs_cherenkov(os.path.join(basedir, fname), run_number, energy, "pion")

    print("=== Processing muon files ===")
    for fname, energy in muon_files.items():
        run_number = fname.split('_')[0].replace("run", "")
        analyze_muon_vs_cherenkov(os.path.join(basedir, fname), run_number, energy, "muon")


if __name__ == "__main__":
    run_all()
