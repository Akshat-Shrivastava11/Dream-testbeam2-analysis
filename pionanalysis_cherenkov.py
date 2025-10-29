import os
import numpy as np
import uproot
import matplotlib.pyplot as plt

# ===================================
# Channels and thresholds
# ===================================
psd         = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"
chrnkov1    = "DRS_Board7_Group2_Channel5"
chrnkov2    = "DRS_Board7_Group2_Channel6"
chrnkov3    = "DRS_Board7_Group2_Channel7"

muon_threshold = 5000
psd_threshold  = 2000
window         = 50
baseline_samples = 20

output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_energywise_overlay"
os.makedirs(output_folder, exist_ok=True)

# ===================================
# Waveform integration
# ===================================
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

# ===================================
# File dictionaries
# ===================================
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'

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

muon_files = {
    'run1447_250927084726.root': '170',
    'run1445_250927074156.root': '110',
}

all_files = {
    "pion": pion_files,
    "positron": positron_files,
    "muon": muon_files,
}

# ===================================
# Group runs by energy
# ===================================
energy_groups = {}
for particle_type, files_dict in all_files.items():
    for fname, energy in files_dict.items():
        energy = str(energy)
        energy_groups.setdefault(energy, []).append((particle_type, fname))

# ===================================
# Process and plot per energy
# ===================================
for energy, runs in sorted(energy_groups.items(), key=lambda x: float(x[0])):
    print(f"\n=== Processing {energy} GeV ===")
    if not runs:
        print(f"⚠️ No runs found for {energy} GeV — skipping.")
        continue

    cher_data = {1: {}, 2: {}, 3: {}}

    print(f"Files found for {energy} GeV:")
    for particle_type, fname in runs:
        print(f"  • {particle_type.upper():<9} → {fname}")

        file_path = os.path.join(basedir, fname)
        beam_label = f"{particle_type.capitalize()} Beam"

        try:
            with uproot.open(file_path) as f:
                tree = f["EventTree"]

                psd_int   = integrate_waveforms(tree[psd].array(library="np"), window, baseline_samples)
                muon_int  = integrate_waveforms(tree[muoncounter].array(library="np"), window, baseline_samples)
                cher1_int = integrate_waveforms(tree[chrnkov1].array(library="np"), window, baseline_samples)
                cher2_int = integrate_waveforms(tree[chrnkov2].array(library="np"), window, baseline_samples)
                cher3_int = integrate_waveforms(tree[chrnkov3].array(library="np"), window, baseline_samples)

            # Apply common cut
            mask = (muon_int < muon_threshold) & (psd_int < psd_threshold)
            print(f"    → Applied cut: (muon_int < {muon_threshold}) & (psd_int < {psd_threshold})")

            # Store Cherenkov integrals
            cher_data[1][beam_label] = cher1_int[mask]
            cher_data[2][beam_label] = cher2_int[mask]
            cher_data[3][beam_label] = cher3_int[mask]

        except Exception as e:
            print(f"    ⚠️ Failed to process {file_path}: {e}")

    # Skip if nothing valid
    if all(len(v) == 0 for v in cher_data.values()):
        print(f"⚠️ No valid Cherenkov data for {energy} GeV — skipping plot.")
        continue

    # ===================================
    # Plot all beam types for this energy
    # ===================================
    cher_labels = ["Cherenkov 1", "Cherenkov 2", "Cherenkov 3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {
        "Pion Beam": "red",
        "Positron Beam": "green",
        "Muon Beam": "blue"
    }

    for i, ax in enumerate(axes, start=1):
        for label, data in cher_data[i].items():
            if len(data) == 0:
                continue
            bins = np.linspace(0, 30000, 100)
            ax.hist(data, bins=bins, histtype='step', linewidth=1.4,
                    label=label, color=colors.get(label, "black"))

        ax.set_yscale("log")
        ax.set_xlabel("ADC")
        ax.set_title(cher_labels[i-1])
        ax.set_xlim(0, 30000)
        ax.set_xticks(np.arange(0, 30001, 5000))
        ax.set_xticks(np.arange(0, 30001, 1000), minor=True)
        ax.tick_params(axis='x', which='major', labelsize=11, length=7, width=1.3)
        ax.tick_params(axis='x', which='minor', length=3, width=0.8)
        ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.3)
        if i == 1:
            ax.set_ylabel("Counts (log scale)")
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle(
        f"Cherenkov Counters — {energy} GeV\nCut: (muon_int < {muon_threshold}) & (psd_int < {psd_threshold})",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    output_file = os.path.join(output_folder, f"CherenkovOverlay_{energy}GeV.pdf")
    plt.savefig(output_file)
    plt.close()
    print(f"✅ Saved overlay plot → {output_file}")

print("\n✅ All per-energy overlay PDFs saved successfully!")
