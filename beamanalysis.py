import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================================================
# Channel definitions
# ======================================================
psd          = "DRS_Board7_Group1_Channel1"
muoncounter  = "DRS_Board7_Group2_Channel4"
chrnkov1     = "DRS_Board7_Group2_Channel5"
chrnkov2     = "DRS_Board7_Group2_Channel6"
chrnkov3     = "DRS_Board7_Group2_Channel7"
trigger      = "DRS_Board7_Group2_Channel1"  # T4 trigger

# ======================================================
# Common waveform integration
# ======================================================
def integrate_waveforms(events, window=50, baseline_samples=20):
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)
        start = max(0, peak_index - window)
        end = min(len(corrected), peak_index + window)
        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)
    return np.array(integrals)

# ======================================================
# Particle classification & stacked histogram
# ======================================================
def stacked_particle_hist(
    file_path, run_number, beam_energy, position="#1",
    window=50, baseline_samples=20,
    muon_threshold=5000, cherenkov_threshold=5000, psd_threshold=4000
):
    print(f"\n=== Processing Run {run_number}, {beam_energy} ===")

    # Load ROOT file
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        psd_events  = tree[psd].array(library="np")
        muon_events = tree[muoncounter].array(library="np")
        cher1       = tree[chrnkov1].array(library="np")
        cher2       = tree[chrnkov2].array(library="np")
        cher3       = tree[chrnkov3].array(library="np")
        t4_events   = tree[trigger].array(library="np")

    # Integrate all channels
    print("Integrating waveforms...")
    psd_int   = integrate_waveforms(psd_events, window, baseline_samples)
    muon_int  = integrate_waveforms(muon_events, window, baseline_samples)
    cher1_int = integrate_waveforms(cher1, window, baseline_samples)
    cher2_int = integrate_waveforms(cher2, window, baseline_samples)
    cher3_int = integrate_waveforms(cher3, window, baseline_samples)
    t4_int    = integrate_waveforms(t4_events, window, baseline_samples)

    # Average Cherenkov signal (or use max)
    cher_combined = np.maximum.reduce([cher1_int, cher2_int, cher3_int])

    # ======================================================
    # Apply classification masks
    # ======================================================
    muon_mask = muon_int > muon_threshold
    pion_mask = (muon_int < muon_threshold) & (cher_combined < cherenkov_threshold) & (psd_int < psd_threshold)
    positron_mask = (muon_int < muon_threshold) & (cher_combined > cherenkov_threshold) & (psd_int > psd_threshold)

    # Extract trigger integrals per particle type
    muon_trig     = t4_int[muon_mask]
    pion_trig     = t4_int[pion_mask]
    positron_trig = t4_int[positron_mask]

    total = len(t4_int)
    muon_frac     = len(muon_trig) / total if total > 0 else 0
    pion_frac     = len(pion_trig) / total if total > 0 else 0
    positron_frac = len(positron_trig) / total if total > 0 else 0

    print(f"Fractions → Muons={muon_frac:.2%}, Pions={pion_frac:.2%}, Positrons={positron_frac:.2%}")

    # ======================================================
    # Plot
    # ======================================================
    bins = np.linspace(0, np.max(t4_int) if len(t4_int) > 0 else 1, 200)
    plt.figure(figsize=(8,5))
    plt.hist([pion_trig, muon_trig, positron_trig],
             bins=bins, stacked=True,
             color=["red", "purple", "blue"], alpha=0.7,
             label=[f"Pions ({pion_frac:.2%})",
                    f"Muons ({muon_frac:.2%})",
                    f"Positrons ({positron_frac:.2%})"])
    plt.hist(t4_int, bins=bins, histtype="step", color="black", linewidth=1.2, label="All events")
    plt.xlabel("T4 (Trigger) integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"T4 integrals — Run {run_number}, {beam_energy}")
    plt.legend()
    plt.tight_layout()

    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Beamanalysis_plots/stacked_fractions"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Run{run_number}_{beam_energy.replace(' ', '_')}.pdf")
    plt.savefig(out_file)
    plt.close()

    print(f"Saved stacked histogram → {out_file}")
    return muon_frac, pion_frac, positron_frac

# ======================================================
# Beam data dictionaries
# ======================================================
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

# ======================================================
# Run all analyses
# ======================================================
for particle_label, files_dict in {
    "positrons": positron_files,
    "pions": pion_files,
    "muons": muon_files
}.items():
    print(f"\n=== Starting {particle_label} runs ===")
    for fname, energy in files_dict.items():
        run_number = fname.split('_')[0].replace("run", "")
        file_path = os.path.join(basedir, fname)
        stacked_particle_hist(
            file_path=file_path,
            run_number=run_number,
            beam_energy=f"{energy}GeV {particle_label}",
            muon_threshold=5000,
            cherenkov_threshold=5000,
            psd_threshold=4000,
        )
