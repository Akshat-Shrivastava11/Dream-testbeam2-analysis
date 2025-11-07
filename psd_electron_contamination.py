import uproot
import matplotlib.pyplot as plt
import numpy as np
import os 



# ======================================================
# Channels
# ======================================================

psd = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"  # <-- muon waveform branch
chrnkov1    = "DRS_Board7_Group2_Channel5"
chrnkov2    = "DRS_Board7_Group2_Channel6"
chrnkov3    = "DRS_Board7_Group2_Channel7"


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
        end   = min(len(corrected), peak_index + window)
        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)
    return np.array(integrals)

# ======================================================
# Original function (unchanged)
# ======================================================
def analyze_electron_contamination(
    file_path, psd, run_number, position, beam_energy,
    window=100, baseline_samples=20
):
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        events = tree[psd].array(library="np")
    print(f"Loaded {len(events)} events from {file_path}")

    integrals = integrate_waveforms(events, window=window, baseline_samples=baseline_samples)
    noise      = integrals[integrals <= 200]
    mips       = integrals[(integrals > 200) & (integrals <= 5000)]
    electrons  = integrals[integrals > 5000]

    total = len(integrals)
    noise_frac = len(noise) / total
    mip_frac   = len(mips) / total
    elec_frac  = len(electrons) / total

    bins = np.linspace(0, max(integrals), 200)
    plt.figure(figsize=(8,5))
    plt.hist([noise, mips, electrons], bins=bins, stacked=True,
             label=[f"Noise ({noise_frac:.2%})",
                    f"MIPs (π, μ) ({mip_frac:.2%})",
                    f"Electrons ({elec_frac:.2%})"],
             color=['gray', 'red', 'blue'], alpha=0.7)
    plt.hist(integrals, bins=bins, histtype="step", color='black', linewidth=1.2, label="All events")
    plt.yscale("log")
    plt.xlabel("Integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Electron counter integrals\nRun {run_number}, {position}, {beam_energy}")
    plt.legend()
    plt.tight_layout()

    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD/stacked_plots_electroncontamination_noisecut500"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Mip_electronfractions_for_{run_number}_energy_{beam_energy.replace(' ', '_')}.pdf")
    plt.savefig(out_file)
    plt.close()

    print(f"Fractions: Noise={noise_frac:.2%}, MIPs={mip_frac:.2%}, Electrons={elec_frac:.2%}")
    print(f"Saved histogram to {out_file}")

    return integrals, noise_frac, mip_frac, elec_frac


# ======================================================
# NEW FUNCTION: same as above but with muon counter cut
# ======================================================
def analyze_electron_contamination_muoncut(
    file_path, psd, run_number, position, beam_energy,
    window=100, baseline_samples=20,
    muoncounter="DRS_Board7_Group2_Channel4", muon_threshold=5000
):
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        psd_events = tree[psd].array(library="np")
        muon_events = tree[muoncounter].array(library="np")

    print(f"Loaded {len(psd_events)} events from {file_path}")

    muon_integrals = integrate_waveforms(muon_events, window=window, baseline_samples=baseline_samples)
    muon_mask = muon_integrals < muon_threshold
    kept_fraction = np.mean(muon_mask)
    print(f"Muon cut applied: keeping {kept_fraction:.2%} of events (threshold={muon_threshold})")

    psd_events_masked = psd_events[muon_mask]
    integrals = integrate_waveforms(psd_events_masked, window=window, baseline_samples=baseline_samples)

    noise      = integrals[integrals <= 200]
    mips       = integrals[(integrals > 200) & (integrals <= 5000)]
    electrons  = integrals[integrals > 5000]

    total = len(integrals)
    noise_frac = len(noise) / total
    mip_frac   = len(mips) / total
    elec_frac  = len(electrons) / total

    bins = np.linspace(0, max(integrals), 200)
    plt.figure(figsize=(8,5))
    plt.hist([noise, mips, electrons], bins=bins, stacked=True,
             label=[f"Noise ({noise_frac:.2%})",
                    f"MIPs (π, μ) ({mip_frac:.2%})",
                    f"Electrons ({elec_frac:.2%})"],
             color=['gray', 'red', 'blue'], alpha=0.7)
    plt.hist(integrals, bins=bins, histtype="step", color='black', linewidth=1.2, label="All events")
    plt.yscale("log")
    plt.xlabel("Integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Electron counter integrals\nRun {run_number}, {position}, {beam_energy}\nMuon cut < {muon_threshold}")
    plt.legend()
    plt.tight_layout()

    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD/stacked_plots_electroncontamination_muoncut"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Mip_electronfractions_muoncut_for_{run_number}_energy_{beam_energy.replace(' ', '_')}.pdf")
    plt.savefig(out_file)
    plt.close()

    print(f"Fractions (with muon cut): Noise={noise_frac:.2%}, MIPs={mip_frac:.2%}, Electrons={elec_frac:.2%}")
    print(f"Saved histogram to {out_file}")

    return integrals, noise_frac, mip_frac, elec_frac


# ======================================================
# Main function: PSD with muon + Cherenkov cuts
# ======================================================
def analyze_electron_contamination_with_muon_cherenkov_cuts(
    file_path, psd, muoncounter, run_number, position, beam_energy,
    window=100, baseline_samples=20,
    cherenkov_cut=5000, muon_cut=5000
):
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        psd_events = tree[psd].array(library="np")
        muon_events = tree[muoncounter].array(library="np")
        cher1 = tree[chrnkov1].array(library="np")
        cher2 = tree[chrnkov2].array(library="np")
        cher3 = tree[chrnkov3].array(library="np")

    print(f"Loaded {len(psd_events)} events from {file_path}")

    # --- Integrate muon and Cherenkov waveforms ---
    muon_integrals = integrate_waveforms(muon_events, window=window, baseline_samples=baseline_samples)
    cher1_ints = integrate_waveforms(cher1, window=window, baseline_samples=baseline_samples)
    cher2_ints = integrate_waveforms(cher2, window=window, baseline_samples=baseline_samples)
    cher3_ints = integrate_waveforms(cher3, window=window, baseline_samples=baseline_samples)

    # --- Apply cuts ---
    muon_mask = muon_integrals < muon_cut
    cherenkov_mask = (cher1_ints < cherenkov_cut) & (cher2_ints < cherenkov_cut) & (cher3_ints < cherenkov_cut)
    combined_mask = muon_mask & cherenkov_mask

    kept_fraction = np.mean(combined_mask)
    print(f"Muon + Cherenkov cuts applied → keeping {kept_fraction:.2%} of events")

    # --- Apply mask to PSD and integrate ---
    psd_events_masked = psd_events[combined_mask]
    integrals = integrate_waveforms(psd_events_masked, window=window, baseline_samples=baseline_samples)

    # --- Split into regions ---
    noise      = integrals[integrals <= 400]
    mips       = integrals[(integrals > 400) & (integrals <= 5000)]
    electrons  = integrals[integrals > 2500]

    total = len(integrals)
    noise_frac = len(noise) / total if total > 0 else 0
    mip_frac   = len(mips) / total if total > 0 else 0
    elec_frac  = len(electrons) / total if total > 0 else 0

    # --- Plot ---
    bins = np.linspace(0, max(integrals) if len(integrals) > 0 else 1, 200)
    plt.figure(figsize=(8,5))
    plt.hist([noise, mips, electrons], bins=bins, stacked=True,
             label=[f"Noise ({noise_frac:.2%})",
                    f"MIPs (π, μ) ({mip_frac:.2%})",
                    f"Electrons ({elec_frac:.2%})"],
             color=['gray', 'red', 'blue'], alpha=0.7)
    plt.hist(integrals, bins=bins, histtype="step", color='black', linewidth=1.2, label="All events")
    plt.yscale("log")
    plt.xlabel("Integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Electron counter integrals\nRun {run_number}, {position}, {beam_energy}\nMuon & Cherenkov < {muon_cut}")
    plt.legend()
    plt.tight_layout()

    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD/stacked_plots_electroncontamination_muon_and_CherenkovCuts"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Mip_electronfractions_muonCherenkovCuts_{run_number}_{beam_energy.replace(' ', '_')}.pdf")
    plt.savefig(out_file)
    plt.close()

    print(f"Fractions (Muon + Cherenkov cuts): Noise={noise_frac:.2%}, MIPs={mip_frac:.2%}, Electrons={elec_frac:.2%}")
    print(f"Saved histogram to {out_file}")

    return integrals, noise_frac, mip_frac, elec_frac

# ======================================================
# Your dataset dictionaries (same)
# ======================================================
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'
positron_files = { 'run1410_250925145231.root': '100', 'run1411_250925154340.root': '120',
                  'run1422_250926102502.root': '110', 'run1409_250925135843.root': '80',
                  'run1412_250925174602.root': '60', 'run1415_250925192957.root': '40',
                  'run1416_250925230347.root': '30', 'run1423_250926105310.root': '20',
                  'run1424_250926124313.root': '10', }
pion_files = { 'run1433_250926213442.root': '120', 'run1432_250926203416.root': '100',
              'run1434_250926222520.root': '160', 'run1429_250926183919.root': '80',
              'run1437_250927003120.root': '60', 'run1438_250927012632.root': '40',
              'run1439_250927023319.root': '30', 'run1441_250927033539.root': '20',
              'run1442_250927050848.root': '10', 'run1452_250927102123.root': '5', }
muon_files = { 'run1447_250927084726.root': '170', 'run1445_250927074156.root': '110' }

# ======================================================
# Modified collect_results: calls both analyses
# ======================================================
def collect_results(files_dict, particle_label):
    results = []
    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")
        print(f"\n=== Processing {fname} ({energy} GeV {particle_label}) ===")

        # Original analysis
        #analyze_electron_contamination(file_path, psd, run_number, "#1", f"{energy}GeV {particle_label}")
        # New analysis with muon cut
        #analyze_electron_contamination_muoncut(file_path, psd, run_number, "#1", f"{energy}GeV {particle_label}",
        #                                       muoncounter=muoncounter, muon_threshold=5000)

        analyze_electron_contamination_with_muon_cherenkov_cuts(file_path=file_path,psd=psd,muoncounter=muoncounter,run_number=run_number,position="#1",
                                                                beam_energy=f"{energy}GeV {particle_label}",muon_cut=5000,cherenkov_cut=5000)
    print("\nAll runs processed for", particle_label)


# ======================================================
# Run all groups
# ======================================================
collect_results(positron_files, "positrons")
collect_results(pion_files, "pions")
collect_results(muon_files, "muons")
