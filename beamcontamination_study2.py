# ===============================
# Beam Contamination Analysis (PSD + muon split)
# ===============================
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Waveform integration
# -------------------------------
def integrate_waveform(event, window=50, baseline_samples=20):
    """Integrate a single waveform around the peak."""
    event_np = np.array(event)
    baseline = np.mean(event_np[:baseline_samples])
    corrected = event_np - baseline
    peak_index = np.argmax(corrected)   # peak for positive pulses
    start, end = max(0, peak_index - window), min(len(corrected), peak_index + window)
    return np.sum(corrected[start:end])

# ===============================
# File paths per beam type
# ===============================
positron_files = {
    'run1527_250929001555.root': '100',
    'run1411_250925154340.root': '120',
    'run1422_250926102502.root': '110',
    'run1409_250925135843.root': '80',
    'run1526_250928235028.root': '60',
    'run1525_250928232144.root': '40',
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

basedir = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT"

# ===============================
# Helper function: stacked hist
# ===============================
def stacked_muon_counter(file_path, muoncounter, psd, run_number, position,
                         beam_energy, window=50, baseline_samples=20, muon_threshold=5000):

    print(f"  → Opening file {file_path}")
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        muon_data = tree[muoncounter].array(library="np")
        psd_data  = tree[psd].array(library="np")
    print(f"  → Loaded {len(muon_data)} muon events and {len(psd_data)} PSD events")

    if len(muon_data) != len(psd_data):
        n = min(len(muon_data), len(psd_data))
        print(f"    ⚠️  Mismatch in event counts (muon={len(muon_data)} , psd={len(psd_data)}). Using first {n}")
        muon_data = muon_data[:n]
        psd_data  = psd_data[:n]

    muon_integrals = np.array([integrate_waveform(ev, window=window, baseline_samples=baseline_samples)
                               for ev in muon_data])
    psd_integrals  = np.array([integrate_waveform(ev, window=50, baseline_samples=baseline_samples)
                               for ev in psd_data])

    total = len(muon_integrals)
    print(f"  → Computed integrals for {total} events")

    # PSD classification
    print("Applying PSD selection masks...")
    noise_mask    = psd_integrals <= 200
    mip_mask      = (psd_integrals > 200) & (psd_integrals <= 5000)
    electron_mask = psd_integrals > 5000

    print(f"Events classified by PSD: Noise={np.sum(noise_mask)}, "
          f"MIPs={np.sum(mip_mask)}, Electrons={np.sum(electron_mask)}")

    # Split MIPs into muons vs pions using muon counter
    pion_mask  = mip_mask & (muon_integrals <= muon_threshold)
    muon_mask  = mip_mask & (muon_integrals >  muon_threshold)

    print(f"Events above muon threshold ({muon_threshold} ADC): {np.sum(muon_mask)}")
    print(f"Events below muon threshold ({muon_threshold} ADC): {np.sum(pion_mask)}")

    # Final categories
    muon_noise = muon_integrals[noise_mask]
    muon_pions = muon_integrals[pion_mask]
    muon_muons = muon_integrals[muon_mask]
    muon_elec  = muon_integrals[electron_mask]

    noise_frac = len(muon_noise) / total if total > 0 else 0
    pion_frac  = len(muon_pions) / total if total > 0 else 0
    muon_frac  = len(muon_muons) / total if total > 0 else 0
    elec_frac  = len(muon_elec)  / total if total > 0 else 0

    print(f"  → Fractions: Noise={noise_frac:.2%}, "
          f"Pions={pion_frac:.2%}, Muons={muon_frac:.2%}, Electrons={elec_frac:.2%}")

    # plot stacked hist
    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/stacked_particlefractions"
    log_out_dir = os.path.join(out_dir, "log")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_out_dir, exist_ok=True)

    bins = np.linspace(0, max(muon_integrals) if total > 0 else 1, 200)
    plt.figure(figsize=(8,5))
    plt.hist([muon_noise, muon_pions, muon_muons, muon_elec],
             bins=bins, stacked=True,
             color=['gray', 'red', 'purple', 'blue'], alpha=0.7,
             label=[f"Noise ({noise_frac:.2%})",
                    f"Pions ({pion_frac:.2%})",
                    f"Muons ({muon_frac:.2%})",
                    f"Electrons ({elec_frac:.2%})"])
    plt.hist(muon_integrals, bins=bins, histtype="step", color='black', linewidth=1.2, label="All events")

    plt.xlabel("Muon Counter Integral (ADC area)")
    plt.ylabel("Events")
    plt.title(f"Run {run_number} ({beam_energy}) - {position}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file_lin = os.path.join(out_dir, f"stacked_muoncounter_run{run_number}_lin.pdf")
    out_file_log = os.path.join(log_out_dir, f"stacked_muoncounter_run{run_number}_log.pdf")
    plt.savefig(out_file_lin)
    plt.yscale("log")
    plt.savefig(out_file_log)
    plt.close()

    return muon_integrals, noise_frac, pion_frac, muon_frac, elec_frac

# ===============================
# Fractions summary
# ===============================
fractions_summary = {
    "positron": {"energy": [], "noise": [], "pion": [], "muon": [], "elec": []},
    "pion":     {"energy": [], "noise": [], "pion": [], "muon": [], "elec": []},
    "muon":     {"energy": [], "noise": [], "pion": [], "muon": [], "elec": []},
}

# ===============================
# Loop over runs
# ===============================
for particle_type, files_dict in all_files.items():
    print(f"\n=== Starting {particle_type} runs ===")
    for i, (fname, energy) in enumerate(files_dict.items(), 1):
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")
        beam_energy = float(energy)

        print(f"\n[{i}/{len(files_dict)}] Processing {particle_type} run {run_number}, energy {beam_energy} GeV ...")
        try:
            muon_integrals, noise_frac, pion_frac, muon_frac, elec_frac = stacked_muon_counter(
                file_path=file_path,
                muoncounter="DRS_Board7_Group2_Channel4",
                psd="DRS_Board7_Group1_Channel1",
                run_number=run_number,
                position="#1",
                beam_energy=f"{beam_energy} GeV {particle_type}",
                window=50,
                baseline_samples=20,
                muon_threshold=5000
            )
            fractions_summary[particle_type]["energy"].append(beam_energy)
            fractions_summary[particle_type]["noise"].append(noise_frac)
            fractions_summary[particle_type]["pion"].append(pion_frac)
            fractions_summary[particle_type]["muon"].append(muon_frac)
            fractions_summary[particle_type]["elec"].append(elec_frac)

            print(f"Run {run_number} done: Noise={noise_frac:.2%}, "
                  f"Pions={pion_frac:.2%}, Muons={muon_frac:.2%}, Electrons={elec_frac:.2%}")
        except Exception as e:
            print(f"Failed to process run {run_number}: {e}")

# ===============================
# Energy vs fraction plots
# ===============================
out_dir_summary = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/stacked_particlefractions/fraction_vs_energy"
os.makedirs(out_dir_summary, exist_ok=True)

for particle_type, data in fractions_summary.items():
    if len(data["energy"]) == 0:
        continue
    energies = np.array(data["energy"])
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    noise = np.array(data["noise"])[sort_idx]
    pions = np.array(data["pion"])[sort_idx]
    muons = np.array(data["muon"])[sort_idx]
    elecs = np.array(data["elec"])[sort_idx]

    plt.figure(figsize=(8,6))
    plt.plot(energies, noise, 'o-', color="gray",   label="Noise")
    plt.plot(energies, pions, 'o-', color="red",    label="Pions")
    plt.plot(energies, muons, 'o-', color="purple", label="Muons")
    plt.plot(energies, elecs, 'o-', color="blue",   label="Electrons")

    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Fraction of events")
    plt.title(f"Fractions vs Beam Energy ({particle_type} runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = os.path.join(out_dir_summary, f"Fractions_vs_Energy_{particle_type}.pdf")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved summary fraction plot for {particle_type} to {out_file}")
