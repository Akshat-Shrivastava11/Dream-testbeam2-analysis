import uproot
import matplotlib.pyplot as plt
import numpy as np
import os 

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
        integrals.append(-area)  # flip sign so negative pulses → positive area
    return np.array(integrals)


def analyze_electron_contamination(
    file_path, psd, run_number, position, beam_energy,
    window=100, baseline_samples=20
):
    # Load events
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        events = tree[psd].array(library="np")

    print(f"Loaded {len(events)} events from {file_path}")

    # Compute integrals
    integrals = integrate_waveforms(events, window=window, baseline_samples=baseline_samples)

    # Classify events
    noise      = integrals[integrals <= 1000]
    mips       = integrals[(integrals > 1000) & (integrals <= 5000)]
    electrons  = integrals[integrals > 5000]

    total_events = len(integrals)
    noise_fraction = len(noise) / total_events
    mip_fraction   = len(mips) / total_events
    electron_fraction = len(electrons) / total_events
    bins = np.linspace(0, max(integrals), 200)
    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(integrals, bins=bins, histtype="step", color='black', label="All events")
    plt.hist(noise, bins=bins, histtype="stepfilled",
            color='gray', alpha=0.5, label=f"Noise ({noise_fraction:.2%})")
    plt.hist(mips, bins=bins, histtype="stepfilled",
            color='red', alpha=0.7, label=f"MIPs (π, μ) ({mip_fraction:.2%})")
    plt.hist(electrons, bins=bins, histtype="stepfilled",
            color='blue', alpha=0.7,
            label=f"Electrons ({electron_fraction:.2%})")

    plt.yscale("log")
    plt.xlabel("Integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Electron counter waveform integrals\nRun {run_number}, {position}, {beam_energy}")
    plt.legend()
    plt.tight_layout()

    # Ensure output directory exists
    out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/plots_electroncontamination"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(
        out_dir,
        f"non_electroncounts_for_{run_number}_energy_{beam_energy.replace(' ', '_')}.pdf"
    )
    plt.savefig(out_file)
    plt.close()

    print(f"Fractions: Noise={noise_fraction:.2%}, MIPs={mip_fraction:.2%}, Electrons={electron_fraction:.2%}")
    print(f"Saved histogram to {out_file}")

    return integrals, noise_fraction, mip_fraction, electron_fraction

basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/' 
positron_files = { 'run1410_250925145231.root': '100', 
                  'run1411_250925154340.root': '120',
                  'run1422_250926102502.root': '110', 
                  'run1409_250925135843.root': '80', 
                  'run1412_250925174602.root': '60', 
                  'run1415_250925192957.root': '40', 
                  'run1416_250925230347.root': '30', 
                  'run1423_250926105310.root': '20', 
                  'run1424_250926124313.root': '10', 
                  } 
pion_files = { 'run1433_250926213442.root': '120', 
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
muon_files = { 'run1447_250927084726.root': '170', 
              'run1445_250927074156.root': '110', 
            }

def collect_results(files_dict, particle_label):
    noise_fractions = []
    mip_fractions = []
    electron_fractions = []
    energys = []
    results = {}

    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")

        integrals, noise_frac, mip_frac, electron_frac = analyze_electron_contamination(
            file_path,
            psd,
            run_number=run_number,
            position="#1",
            beam_energy=f"{energy}GeV {particle_label}"
        )

        noise_fractions.append(noise_frac)
        mip_fractions.append(mip_frac)
        electron_fractions.append(electron_frac)
        energys.append(int(energy))

        results[run_number] = {
            "energy": energy,
            "integrals": integrals,
            "noise_fraction": noise_frac,
            "mip_fraction": mip_frac,
            "electron_fraction": electron_frac
        }

    energies = np.array(energys)
    noise_fractions = np.array(noise_fractions)
    mip_fractions = np.array(mip_fractions)
    electron_fractions = np.array(electron_fractions)

    sort_idx = np.argsort(energies)
    return (energies[sort_idx],
            noise_fractions[sort_idx],
            mip_fractions[sort_idx],
            electron_fractions[sort_idx],
            results)


# === Example usage with your dicts ===
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'

energies_pos, noise_pos, mip_pos, elec_pos, results_pos = collect_results(positron_files, "positrons")
energies_pi, noise_pi, mip_pi, elec_pi, results_pi = collect_results(pion_files, "pions")
energies_mu, noise_mu, mip_mu, elec_mu, results_mu = collect_results(muon_files, "muons")


# --- Individual plots ---
def plot_fractions(energies, noise, mip, elec, beam_name, save_name):
    plt.figure(figsize=(8,6))
    plt.plot(energies, elec, marker='o', linestyle='-', color='blue', label="Electrons")
    plt.plot(energies, mip, marker='s', linestyle='--', color='red', label="MIPs")
    plt.plot(energies, noise, marker='^', linestyle=':', color='gray', label="Noise")

    plt.xlabel("Beam Energy [GeV]", fontsize=14)
    plt.ylabel("Fraction of events", fontsize=14)
    plt.title(f"Fractions vs Beam Energy ({beam_name} beam)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

# Individual plots
plot_fractions(energies_pos, noise_pos, mip_pos, elec_pos, "Positron", "fractions_vs_energy_positron.pdf")
plot_fractions(energies_pi, noise_pi, mip_pi, elec_pi, "Pion", "fractions_vs_energy_pion.pdf")
plot_fractions(energies_mu, noise_mu, mip_mu, elec_mu, "Muon", "fractions_vs_energy_muon.pdf")


# --- Combined plot ---
plt.figure(figsize=(9,7))

# Electrons
plt.plot(energies_pos, elec_pos, marker='o', linestyle='-', color='blue', label="Electrons (positron beam)")
plt.plot(energies_pi, elec_pi, marker='o', linestyle='--', color='blue', alpha=0.7, label="Electrons (pion beam)")
plt.plot(energies_mu, elec_mu, marker='o', linestyle=':', color='blue', alpha=0.7, label="Electrons (muon beam)")

# MIPs
plt.plot(energies_pos, mip_pos, marker='s', linestyle='-', color='red', label="MIPs (positron beam)")
plt.plot(energies_pi, mip_pi, marker='s', linestyle='--', color='red', alpha=0.7, label="MIPs (pion beam)")
plt.plot(energies_mu, mip_mu, marker='s', linestyle=':', color='red', alpha=0.7, label="MIPs (muon beam)")

# Noise
plt.plot(energies_pos, noise_pos, marker='^', linestyle='-', color='gray', label="Noise (positron beam)")
plt.plot(energies_pi, noise_pi, marker='^', linestyle='--', color='gray', alpha=0.7, label="Noise (pion beam)")
plt.plot(energies_mu, noise_mu, marker='^', linestyle=':', color='gray', alpha=0.7, label="Noise (muon beam)")

plt.xlabel("Beam Energy [GeV]", fontsize=14)
plt.ylabel("Fraction of events", fontsize=14)
plt.title("Fractions vs Beam Energy (All beams)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(ncol=2, fontsize=10)  # compact legend
plt.tight_layout()
plt.savefig("fractions_vs_energy_allbeams.pdf")
plt.close()
