import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Trigger branches ---
KT1trigger = "DRS_Board7_Group2_Channel2"
KT2trigger = "DRS_Board7_Group2_Channel3"

# --- Waveform integration ---
def integrate_waveforms(events, window=50, baseline_samples=20):
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)

        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)

        area = np.trapz(corrected[start:end], dx=1)
        integrals.append(-area)  # flip sign → positive
    return np.array(integrals)

# --- Efficiency ---
def compute_efficiency(integrals, threshold):
    if len(integrals) == 0:
        return 0.0
    return np.sum(integrals > threshold) / len(integrals)

# --- Trigger analysis ---
def analyze_trigger_efficiency(file_path, trigger_branch, window=100, baseline_samples=20):
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        events = tree[trigger_branch].array(library="np")
    return integrate_waveforms(events, window=window, baseline_samples=baseline_samples)

# --- File definitions ---
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/' 

positron_files = { 
    'run1422_250926102502.root': '110',
    'run1527_250929001555.root': '100',
    'run1526_250928235028.root': '60',
    'run1525_250928232144.root': '40',
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

# --- Analysis ---
threshold = 1000
results = {ptype: {"energy": [], "KT1_eff": [], "KT2_eff": []} for ptype in all_files}

for particle_type, files_dict in all_files.items():
    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")

        print(f"Processing {particle_type} run {run_number}, {energy} GeV...")
        try:
            # KT1
            integrals_KT1 = analyze_trigger_efficiency(file_path, KT1trigger)
            eff_KT1 = compute_efficiency(integrals_KT1, threshold)

            # KT2
            integrals_KT2 = analyze_trigger_efficiency(file_path, KT2trigger)
            eff_KT2 = compute_efficiency(integrals_KT2, threshold)

            # Save results
            results[particle_type]["energy"].append(float(energy))
            results[particle_type]["KT1_eff"].append(eff_KT1)
            results[particle_type]["KT2_eff"].append(eff_KT2)

            print(f"   ↳ KT1 eff={eff_KT1:.2%}, KT2 eff={eff_KT2:.2%} ({len(integrals_KT1)} events)")

            # --- Plot integrals distribution comparison ---
            plt.figure(figsize=(7,5))
            plt.hist(integrals_KT1, bins=100, histtype="step", label=f"KT1 eff={eff_KT1:.2%}", color="blue")
            plt.hist(integrals_KT2, bins=100, histtype="step", label=f"KT2 eff={eff_KT2:.2%}", color="green")
            plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
            plt.xlabel("Integrated ADC (area)")
            plt.ylabel("Events")
            plt.title(f"{particle_type.capitalize()} Run {run_number} ({energy} GeV)\nKT1 vs KT2 integrals")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()

            out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/KT_trigger_plots/"
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{particle_type}_run{run_number}_{energy}_KT1vsKT2_integrals.pdf")
            plt.savefig(out_file)
            plt.close()
            print(f"   ↳ Saved KT1 vs KT2 integral histogram: {out_file}")

        except Exception as e:
            print(f"⚠️ Failed on run {run_number}: {e}")

# --- Plot Efficiency vs. Energy ---
plt.figure(figsize=(8,6))
for ptype, marker, color in zip(results.keys(), ["o", "s", "^"], ["red", "blue", "green"]):
    energies = np.array(results[ptype]["energy"])
    effs_KT1 = np.array(results[ptype]["KT1_eff"])
    effs_KT2 = np.array(results[ptype]["KT2_eff"])
    order = np.argsort(energies)

    plt.plot(energies[order], effs_KT1[order], marker=marker, linestyle="-", color=color, label=f"{ptype.capitalize()} KT1")
    plt.plot(energies[order], effs_KT2[order], marker=marker, linestyle="--", color=color, label=f"{ptype.capitalize()} KT2")

plt.xlabel("Beam Energy [GeV]")
plt.ylabel(f"Trigger Efficiency (thr = {threshold})")
plt.title("KT1 vs KT2 Trigger Efficiency vs Beam Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()

out_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/KT_trigger_plots"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f"KT1vsKT2_eff_vs_energy_thr{threshold}.pdf")
plt.savefig(out_file)
plt.close()
print(f"✅ Saved KT1 vs KT2 efficiency vs energy plot to {out_file}")
