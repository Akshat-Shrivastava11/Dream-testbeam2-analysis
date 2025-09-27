import uproot
import matplotlib.pyplot as plt
import numpy as np

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

    out_file = f"Muoncounts_for_{run_number}_energy_{beam_energy.replace(' ', '_')}.pdf"
    plt.savefig(out_file)
    plt.close()

    print(f"Muon contamination = {muon_contamination:.2%}")
    print(f"Saved histogram to {out_file}")

    return integrals, muon_contamination


# Example usage:
# file_path = "data_testbeam/testbeam_round2/run1355_250924165834_converted.root"
file_path = "data_testbeam/testbeam_round2/run1355_250924165834_converted.root"
analyze_muon_contamination(file_path, muoncounter, run_number="1355", position="#1", beam_energy="80GeV positrons")
