import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# ===================================
# Channels
# ===================================
psd         = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"

window           = 50
baseline_samples = 20
muon_threshold   = 5000  # Muon veto threshold

output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD/Integrals_shape_postmuonveto_overlay"
os.makedirs(output_folder, exist_ok=True)

# ===================================
# Waveform integration
# ===================================
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

# ===================================
# Beam files
# ===================================
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'

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

muon_files = {
    'run1447_250927084726.root': '170',
    'run1445_250927074156.root': '110',
}

all_files = {
    "pion": pion_files,
    "positron": positron_files,
    "muon": muon_files,
}

# ===================================
# Collect PSD integrals by energy
# ===================================
energy_dict = {}

for particle_type, files_dict in all_files.items():
    print(f"\n=== Processing {particle_type} runs ===")
    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")
        print(f"Run {run_number}, {energy} GeV {particle_type}")

        try:
            with uproot.open(file_path) as f:
                tree = f["EventTree"]
                psd_int  = integrate_waveforms(tree[psd].array(library="np"), window, baseline_samples)
                muon_int = integrate_waveforms(tree[muoncounter].array(library="np"), window, baseline_samples)

            # Muon veto
            mask = muon_int < muon_threshold
            psd_int = psd_int[mask]

            # Store by energy
            energy_dict.setdefault(energy, {}).setdefault(particle_type, []).append(psd_int)

        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

# ===================================
# Overlay plots by matching energies
# ===================================
bins = np.linspace(0, 30000, 60)

for energy, data in energy_dict.items():
    plt.figure(figsize=(8,6))
    plt.title(f"PSD Integrals Overlayed (Muon Veto) — {energy} GeV")
    plt.xlabel("PSD Integral (ADC·ns)")
    plt.ylabel("Counts")
    plt.xlim(0, 30000)
    plt.grid(True, linestyle='--', alpha=0.3)

    for particle_type, psd_lists in data.items():
        psd_all = np.concatenate(psd_lists)
        plt.hist(psd_all, bins=bins, histtype='step', linewidth=2, label=f"{particle_type} (N={len(psd_all)})")

    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 minor ticks per major tick
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', length=3)
    ax.tick_params(axis='y', which='minor', length=3)
    plt.tight_layout()
    outpath = os.path.join(output_folder, f"PSD_overlay_{energy}GeV.pdf")
    plt.savefig(outpath)
    plt.close()
    print(f"✅ Saved overlay plot → {outpath}")
