import uproot
import matplotlib.pyplot as plt
import numpy as np
import os 

muoncounter = "DRS_Board7_Group2_Channel4"
def integrate_waveforms(events, window=100, baseline_samples=20):
    """
    Compute baseline-corrected integrals for each waveform in events.

    Parameters
    ----------
    events : list of arrays
        Each element is an event (array of waveform samples).
    window : int, optional
        Number of samples on each side of the peak to integrate (default: 100).
    baseline_samples : int, optional
        Number of samples at the start used for baseline calculation (default: 20).

    Returns
    -------
    integrals : ndarray
        Array of integrated (positive) areas for each waveform.
    """
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)

        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)

        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)  # flip sign so negative pulses → positive area

    return np.array(integrals)


def analyze_muon_contamination(
    file_path, muoncounter, run_number, position, beam_energy,
    window=100, baseline_samples=20, threshold=5000
):
    """
    Load a ROOT file, compute waveform integrals, and make histogram with muon contamination.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    muoncounter : str
        Branch name for the muon counter waveform.
    run_number : str
        Run identifier for labeling plots.
    position : str
        Detector position label.
    beam_energy : str
        Beam energy and particle type string for labeling.
    window : int
        Integration window half-width around peak (default: 100).
    baseline_samples : int
        Number of samples for baseline estimation (default: 20).
    threshold : float
        Integral threshold for muon identification (default: 5000).
    """
    # Load events
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        events = tree[muoncounter].array(library="np")

    print(f"Loaded {len(events)} events from {file_path}")

    # Compute integrals
    integrals = integrate_waveforms(events, window=window, baseline_samples=baseline_samples)

    # Separate noise vs muons
    noise = integrals[integrals <= threshold]
    muons = integrals[integrals > threshold]

    muon_contamination = len(muons) / len(integrals)

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(noise, bins=1000, histtype="stepfilled",
             color='gray', alpha=0.5, label="Noise")
    plt.hist(muons, bins=1000, histtype="stepfilled",
             color='blue', alpha=0.7,
             label=f"Muons, contamination = {muon_contamination:.2%}")

    plt.xlabel("Integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Muon counter waveform integrals\nRun {run_number}, {position}, {beam_energy}")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    out_file = f"/lustre/research/hep/akshriva/Dream-testbeam2-analysis/plots_muoncontamination/Muoncounts_for_{run_number}_energy_{beam_energy.replace(' ', '_')}.pdf"
    plt.savefig(out_file)
    plt.close()

    print(f"Muon contamination = {muon_contamination:.2%}")
    print(f"Saved histogram to {out_file}")

    return integrals, muon_contamination


# Example usage:
# file_path = "data_testbeam/testbeam_round2/run1355_250924165834_converted.root"
#file_path = "data_testbeam/testbeam_round2/run1355_250924165834_converted.root"


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
}


muon_files = {
    'run1447_250927084726.root': '170',
    'run1445_250927074156.root': '110',
}


def collect_results(files_dict, particle_label):
    muon_contaminations = []
    energys = []
    results = {}

    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)

        run_number = fname.split('_')[0].replace("run", "")

        integrals, muon_contamination = analyze_muon_contamination(
            file_path,
            muoncounter,
            run_number=run_number,
            position="#1",
            beam_energy=f"{energy}GeV {particle_label}"
        )

        muon_contaminations.append(muon_contamination)
        energys.append(int(energy))
        results[run_number] = {
            "energy": energy,
            "integrals": integrals,
            "muon_contamination": muon_contamination
        }

    energies = np.array(energys)
    contaminations = np.array(muon_contaminations)

    sort_idx = np.argsort(energies)
    return energies[sort_idx], contaminations[sort_idx], results

# Collect for both
energies_pos, cont_pos, results_pos = collect_results(positron_files, "positrons")
energies_pi, cont_pi, results_pi = collect_results(pion_files, "pions")
energies_mu, cont_mu, results_mu = collect_results(muon_files, "muons")
# Plot together
plt.figure(figsize=(8,6))
plt.plot(energies_pos, cont_pos, marker='o', linestyle='-', color='g', label="Positrons")
plt.plot(energies_pi, cont_pi, marker='s', linestyle='--', color='r', label="Pions")
plt.plot(energies_mu, cont_mu, marker='^', linestyle=':', color='b', label="Muons")
plt.xlabel("Beam Energy [GeV]", fontsize=14)
plt.ylabel("Muon Contamination ", fontsize=14)
plt.title("Muon Contamination vs Beam Energy", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("muoncontamination_testbeam02_pos_&_pi.pdf")
