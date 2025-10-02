import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

muoncounter = "DRS_Board7_Group2_Channel4"
psd = "DRS_Board7_Group1_Channel1"

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

def plot_max_muon_noise_and_signal(file_path, muoncounter, psd, run_number, beam_energy,
                                   window=50, baseline_samples=20, psd_threshold=200):
    """
    Find the event with maximum muon counter integral among PSD-noise events
    and among PSD-signal events, then plot both waveforms (muon counter + PSD).
    """
    print(f"\n=== Plotting max muon (PSD noise & signal) events for Run {run_number}, {beam_energy} ===")
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        muon_events = tree[muoncounter].array(library="np")
        psd_events  = tree[psd].array(library="np")

    # Compute integrals
    muon_integrals = integrate_waveforms(muon_events, window=window, baseline_samples=baseline_samples)
    psd_integrals  = integrate_waveforms(psd_events, window=50, baseline_samples=baseline_samples)

    # Masks
    noise_mask  = psd_integrals <= psd_threshold
    signal_mask = psd_integrals > psd_threshold

    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD&MUONmax_muon_noise_waveforms"
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for case_name, mask in [("Noise", noise_mask), ("Signal", signal_mask)]:
        if np.sum(mask) == 0:
            print(f"⚠️ No PSD {case_name.lower()} events found in this run!")
            continue

        muon_masked_integrals = muon_integrals[mask]
        muon_masked_events    = muon_events[mask]
        psd_masked_events     = psd_events[mask]

        # Find max muon integral event in this subset
        max_idx_local = np.argmax(muon_masked_integrals)
        max_val = muon_masked_integrals[max_idx_local]

        global_indices = np.where(mask)[0]
        max_idx_global = global_indices[max_idx_local]

        print(f"Max muon counter (PSD {case_name}) integral = {max_val:.2f} (event {max_idx_global})")

        # Waveforms
        muon_waveform = muon_masked_events[max_idx_local]
        psd_waveform  = psd_masked_events[max_idx_local]

        # Baseline subtract
        muon_baseline = np.mean(muon_waveform[:baseline_samples])
        psd_baseline  = np.mean(psd_waveform[:baseline_samples])
        muon_corrected = muon_waveform - muon_baseline
        psd_corrected  = psd_waveform - psd_baseline

        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(muon_corrected, label="Muon counter waveform", color="blue")
        plt.plot(psd_corrected, label="PSD waveform", color="red")
        plt.xlabel("Sample")
        plt.ylabel("ADC (baseline subtracted)")
        plt.title(f"Max muon event (PSD {case_name})\nRun {run_number}, {beam_energy}")
        plt.legend()
        plt.tight_layout()

        out_file = os.path.join(out_dir, f"MaxMuon{case_name}_Run{run_number}_{beam_energy.replace(' ', '_')}.pdf")
        plt.savefig(out_file)
        plt.close()
        print(f"Saved max muon PSD-{case_name.lower()} waveform plot to {out_file}")

        results[case_name] = (max_idx_global, max_val)

    return results



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

# --- Combined dictionary ---
all_files = {
    "positron": positron_files,
    "pion": pion_files,
    "muon": muon_files,
}

# --- Loop over all files ---
for particle_type, files_dict in all_files.items():
    print(f"\n=== Starting {particle_type} runs ===")
    for i, (fname, energy) in enumerate(files_dict.items(), 1):
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")
        beam_energy = f"{energy}GeV {particle_type}"

        print(f"\n[{i}/{len(files_dict)}] Processing {particle_type} run {run_number}, energy {energy}GeV ...")
        try:
            results = plot_max_muon_noise_and_signal(
                file_path=file_path,
                muoncounter=muoncounter,
                psd=psd,
                run_number=run_number,
                beam_energy=beam_energy,
                window=50,
                baseline_samples=20
                
            )

            print(f"Done run {run_number}: max muon integral = {max_val:.2f} (event {max_idx})")
        except Exception as e:
            print(f"Failed to process run {run_number}: {e}")
