# ===============================
# Beam Contamination Analysis
# ===============================
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Branches and threshold
# -------------------------------
MUON_BRANCH = "DRS_Board7_Group2_Channel4"
PSD_BRANCH = "DRS_Board7_Group1_Channel1"
MUON_THRESHOLD = 5000

# -------------------------------
# Waveform integration
# -------------------------------
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
        integrals.append(-area)  # make positive
    return np.array(integrals)

# -------------------------------
# Classification
# -------------------------------
def classify_events(psd_integrals, muon_integrals):
    # PSD selection
    noise_mask = psd_integrals <= 200
    mip_mask   = (psd_integrals > 200) & (psd_integrals <= 5000)
    electron_mask = psd_integrals > 5000

    # Muons: MIP events with muon_integral > threshold
    muon_mask = mip_mask & (muon_integrals > MUON_THRESHOLD)
    pion_mask = mip_mask & (muon_integrals <= MUON_THRESHOLD)

    return {
        "noise": np.sum(noise_mask),
        "pions": np.sum(pion_mask),
        "muons": np.sum(muon_mask),
        "electrons": np.sum(electron_mask),
        "total": len(psd_integrals)
    }

# -------------------------------
# Process files
# -------------------------------
def process_files(file_dict, basedir):
    results = {}
    for fname, energy in file_dict.items():
        file_path = os.path.join(basedir, fname)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with uproot.open(file_path) as f:
            tree = f["EventTree"]
            muon_integrals = integrate_waveforms(tree[MUON_BRANCH].array(library="np"))
            psd_integrals = integrate_waveforms(tree[PSD_BRANCH].array(library="np"))

        fractions = classify_events(psd_integrals, muon_integrals)
        # compute fractions
        for k in ["noise","pions","muons","electrons"]:
            fractions[k] /= fractions["total"]
        results[float(energy)] = fractions
        print(f"Processed {fname} ({energy} GeV) -> {fractions}")

    return results

# -------------------------------
# Plot energy vs fraction line
# -------------------------------
def plot_energy_vs_fraction(results, beam_name, outdir):
    os.makedirs(outdir, exist_ok=True)
    energies = sorted(results.keys())
    noise = [results[e]["noise"] for e in energies]
    pions = [results[e]["pions"] for e in energies]
    muons = [results[e]["muons"] for e in energies]
    electrons = [results[e]["electrons"] for e in energies]

    plt.figure(figsize=(8,6))
    plt.plot(energies, noise, 'o-', color="gray", label="Noise")
    plt.plot(energies, pions, 's-', color="red", label="Pions")
    plt.plot(energies, muons, 'd-', color="purple", label="Muons")
    plt.plot(energies, electrons, '^-', color="blue", label="Electrons")
    plt.xlabel("Beam Energy [GeV]")
    plt.ylabel("Fraction of Events")
    plt.title(f"{beam_name} Beam: Energy vs Fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{beam_name.lower()}_energy_vs_fraction.png"))
    plt.close()

# -------------------------------
# Plot stacked bar chart
# -------------------------------
def plot_stacked_fractions(results, beam_name, outdir):
    os.makedirs(outdir, exist_ok=True)
    energies = sorted(results.keys())
    noise = [results[e]["noise"] for e in energies]
    pions = [results[e]["pions"] for e in energies]
    muons = [results[e]["muons"] for e in energies]
    electrons = [results[e]["electrons"] for e in energies]

    bottom1 = np.array(noise)
    bottom2 = bottom1 + np.array(pions)
    bottom3 = bottom2 + np.array(muons)

    plt.figure(figsize=(8,6))
    plt.bar(energies, noise, width=5, color="gray", label="Noise")
    plt.bar(energies, pions, width=5, bottom=bottom1, color="red", label="Pions")
    plt.bar(energies, muons, width=5, bottom=bottom2, color="purple", label="Muons")
    plt.bar(energies, electrons, width=5, bottom=bottom3, color="blue", label="Electrons")

    plt.xlabel("Beam Energy [GeV]")
    plt.ylabel("Fraction of Events")
    plt.title(f"{beam_name} Beam: Composition by Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{beam_name.lower()}_stacked_energy_fractions.png"))
    plt.close()

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    basedir = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/"

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

    outdir = "./beam_fraction_plots"

    for beam_name, file_dict in [("Positron", positron_files),
                                 ("Pion", pion_files),
                                 ("Muon", muon_files)]:
        print(f"\n=== Processing {beam_name} Beam ===")
        results = process_files(file_dict, basedir)
        if results:
            plot_energy_vs_fraction(results, beam_name, outdir)
            plot_stacked_fractions(results, beam_name, outdir)
            print(f"Saved plots for {beam_name} beam in {outdir}")

