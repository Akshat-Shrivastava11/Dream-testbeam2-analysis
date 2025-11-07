# beamcontamination (PSD integrals stacked, muon counter overlay + summary plots)
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# Channels
muoncounter = "DRS_Board7_Group2_Channel4"
psd = "DRS_Board7_Group1_Channel1"

def integrate_waveforms(events, window=50, baseline_samples=20):
    """Integrate waveform area around the peak (baseline subtracted)."""
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

def plot_psd_with_muons(file_path, run_number, beam_energy,
                        window=50, baseline_samples=20, muon_threshold=5000):
    """Plot PSD integrals with muon-tagged events stacked on top."""
    print(f"\n=== Processing Run {run_number}, {beam_energy} ===")

    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        psd_events  = tree[psd].array(library="np")
        muon_events = tree[muoncounter].array(library="np")

    # Integrals
    psd_integrals  = integrate_waveforms(psd_events, window=window, baseline_samples=baseline_samples)
    muon_integrals = integrate_waveforms(muon_events, window=window, baseline_samples=baseline_samples)

    # Muon-tagged events (mask from muon counter)
    muon_mask = muon_integrals > muon_threshold

    # --- Plotting ---
    bins = np.linspace(min(psd_integrals), max(psd_integrals), 150)

    # Log scale
    plt.figure(figsize=(8,5))
    plt.hist(psd_integrals, bins=bins, color="gray", alpha=0.6, label="All PSD events")
    plt.hist(psd_integrals[muon_mask], bins=bins, color="blue", alpha=0.7, label="Muon-tagged")
    plt.yscale("log")
    plt.xlabel("PSD integrated ADC (area)")
    plt.ylabel("Counts")
    plt.title(f"PSD integrals with muon contamination\nRun {run_number}, {beam_energy}")
    plt.legend()
    plt.tight_layout()
    out_dir_log = "out_psd_muon_log"
    os.makedirs(out_dir_log, exist_ok=True)
    plt.savefig(os.path.join(out_dir_log, f"PSD_muon_Run{run_number}_{beam_energy}.pdf"))
    plt.close()

    # Linear scale
    plt.figure(figsize=(8,5))
    plt.hist(psd_integrals, bins=bins, color="gray", alpha=0.6, label="All PSD events")
    plt.hist(psd_integrals[muon_mask], bins=bins, color="blue", alpha=0.7, label="Muon-tagged")
    plt.xlabel("PSD integrated ADC (area)")
    plt.ylabel("Counts")
    plt.title(f"PSD integrals with muon contamination\nRun {run_number}, {beam_energy}")
    plt.legend()
    plt.tight_layout()
    out_dir_lin = "out_psd_muon_linear"
    os.makedirs(out_dir_lin, exist_ok=True)
    plt.savefig(os.path.join(out_dir_lin, f"PSD_muon_Run{run_number}_{beam_energy}.pdf"))
    plt.close()

    contamination_frac = np.sum(muon_mask) / len(muon_mask)
    return contamination_frac

def plot_summary(energies, contaminations, beam_name):
    """Summary: contamination fraction vs beam energy."""
    plt.figure(figsize=(8,6))
    plt.plot(energies, contaminations, marker="o", linestyle="-", color="blue")
    plt.xlabel("Beam Energy [GeV]", fontsize=14)
    plt.ylabel("Muon contamination fraction", fontsize=14)
    plt.title(f"Muon contamination vs Beam Energy ({beam_name} beam)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_dir = "out_summary"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"muon_contam_vs_energy_{beam_name}.pdf"))
    plt.close()

# --- File paths and energies ---
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'

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


all_files = {
    "positron": positron_files,
    "pion": pion_files,
    "muon": muon_files,
}

# --- Loop over all files and collect contamination fractions ---
for particle_type, files_dict in all_files.items():
    print(f"\n=== Starting {particle_type} runs ===")
    energies = []
    contaminations = []
    for i, (fname, energy) in enumerate(files_dict.items(), 1):
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")
        beam_energy = f"{energy}GeV {particle_type}"

        try:
            cont_frac = plot_psd_with_muons(file_path, run_number, beam_energy)
            energies.append(int(energy))
            contaminations.append(cont_frac)
            print(f"Finished Run {run_number}: contamination={cont_frac:.2%}")
        except Exception as e:
            print(f"Failed Run {run_number}: {e}")

    # Make summary plot per beam type
    if energies:
        sort_idx = np.argsort(energies)
        plot_summary(np.array(energies)[sort_idx], np.array(contaminations)[sort_idx], particle_type)
