#beamcontamination
import uproot
import matplotlib.pyplot as plt
import numpy as np

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

def stacked_muon_counter(file_path, muoncounter, psd, run_number, position, beam_energy,
                         window=100, baseline_samples=20, muon_threshold=5000):
    """Make a stacked plot of muon counter events using PSD selection with verbose prints."""

    print(f"\n=== Processing Run {run_number}, {beam_energy} ===")
    print(f"Opening file: {file_path}")

    # Load ROOT events
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        muon_events = tree[muoncounter].array(library="np")
        psd_events   = tree[psd].array(library="np")
    print(f"Loaded {len(muon_events)} events from muon counter")
    print(f"Loaded {len(psd_events)} events from PSD channel")

    # Compute integrals
    print("Computing muon counter integrals...")
    muon_integrals = integrate_waveforms(muon_events, window=window, baseline_samples=baseline_samples)
    print("Computing PSD integrals...")
    psd_integrals   = integrate_waveforms(psd_events, window=50, baseline_samples=baseline_samples)

    print("Applying PSD selection masks...")
    noise_mask      = psd_integrals <= 200
    mip_mask        = (psd_integrals > 200) & (psd_integrals <= 5000)
    electron_mask   = psd_integrals > 5000

    print(f"Events classified by PSD: Noise={np.sum(noise_mask)}, MIPs={np.sum(mip_mask)}, Electrons={np.sum(electron_mask)}")

    # Muon counter selection
    muon_mask = muon_integrals > muon_threshold
    print(f"Events above muon threshold ({muon_threshold} ADC): {np.sum(muon_mask)}")

    # Separate muon counter integrals by PSD category
    muon_noise = muon_integrals[noise_mask]
    muon_mip   = muon_integrals[mip_mask]
    muon_elec  = muon_integrals[electron_mask]

    # Fractions for legend
    total_events = len(muon_integrals)
    noise_frac = len(muon_noise) / total_events
    mip_frac   = len(muon_mip) / total_events
    elec_frac  = len(muon_elec) / total_events
    print(f"Fractions: Noise={noise_frac:.2%}, MIPs={mip_frac:.2%}, Electrons={elec_frac:.2%}")

    # --- Log-scale histogram ---
    print("Generating stacked histogram (log scale)...")
    bins = np.linspace(0, max(muon_integrals), 200)
    plt.figure(figsize=(8,5))
    plt.hist([muon_noise, muon_mip, muon_elec], bins=bins, stacked=True,
            color=['gray', 'red', 'blue'], alpha=0.7,
            label=[f"Noise ({noise_frac:.2%})",
                    f"MIPs (π, μ) ({mip_frac:.2%})",
                    f"Electrons ({elec_frac:.2%})"])
    plt.hist(muon_integrals, bins=bins, histtype="step", color='black', linewidth=1.2, label="All events")
    plt.yscale("log")
    plt.xlabel("Muon counter integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Muon counter waveform integrals with PSD selection\nRun {run_number}, {position}, {beam_energy}")
    plt.legend()
    plt.tight_layout()

    out_dir_log = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/stacked_plots_muon"
    os.makedirs(out_dir_log, exist_ok=True)
    out_file_log = os.path.join(out_dir_log, f"Muon_counter_PSDstack_Run{run_number}_{beam_energy.replace(' ', '_')}.pdf")
    plt.savefig(out_file_log)
    plt.close()
    print(f"Saved log-scale histogram to {out_file_log}")

    # --- Linear-scale histogram ---
    print("Generating stacked histogram (linear scale)...")
    plt.figure(figsize=(8,5))
    plt.hist([muon_noise, muon_mip, muon_elec], bins=bins, stacked=True,
            color=['gray', 'red', 'blue'], alpha=0.7,
            label=[f"Noise ({noise_frac:.2%})",
                    f"MIPs (π, μ) ({mip_frac:.2%})",
                    f"Electrons ({elec_frac:.2%})"])
    plt.hist(muon_integrals, bins=bins, histtype="step", color='black', linewidth=1.2, label="All events")
    #plt.yscale("log")  # linear scale
    plt.xlabel("Muon counter integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Muon counter waveform integrals with PSD selection\nRun {run_number}, {position}, {beam_energy}")
    plt.legend()
    plt.tight_layout()

    out_dir_lin = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/LogOff_stacked_plots_muon"
    os.makedirs(out_dir_lin, exist_ok=True)
    out_file_lin = os.path.join(out_dir_lin, f"Muon_counter_PSDstack_Run{run_number}_{beam_energy.replace(' ', '_')}.pdf")
    plt.savefig(out_file_lin)
    plt.close()
    print(f"Saved linear-scale histogram to {out_file_lin}")


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

# --- Combined dictionary for convenience ---
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
            muon_integrals, noise_frac, mip_frac, elec_frac = stacked_muon_counter(
                file_path=file_path,
                muoncounter="DRS_Board7_Group2_Channel4",
                psd="DRS_Board7_Group1_Channel1",
                run_number=run_number,
                position="#1",
                beam_energy=beam_energy,
                window=50,
                baseline_samples=20,
                muon_threshold=5000
            )
            print(f"Run {run_number} done: Noise={noise_frac:.2%}, MIPs={mip_frac:.2%}, Electrons={elec_frac:.2%}")
        except Exception as e:
            print(f"Failed to process run {run_number}: {e}")

