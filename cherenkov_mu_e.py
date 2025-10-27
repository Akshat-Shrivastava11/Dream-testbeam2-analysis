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
psd_threshold  = 4000
window         = 50
baseline_samples = 20

output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_mu_e_plots"
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
# Beam files
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
# Loop over all files
# ===================================
for particle_type, files_dict in all_files.items():
    print(f"\n=== Processing {particle_type} runs ===")
    for fname, energy in files_dict.items():
        file_path = os.path.join(basedir, fname)
        run_number = fname.split('_')[0].replace("run", "")
        beam_energy = f"{energy}GeV_{particle_type}"

        print(f"Run {run_number}, {beam_energy}")

        try:
            with uproot.open(file_path) as f:
                tree = f["EventTree"]
                psd_int   = integrate_waveforms(tree[psd].array(library="np"), window, baseline_samples)
                muon_int  = integrate_waveforms(tree[muoncounter].array(library="np"), window, baseline_samples)
                cher1_int = integrate_waveforms(tree[chrnkov1].array(library="np"), window, baseline_samples)
                cher2_int = integrate_waveforms(tree[chrnkov2].array(library="np"), window, baseline_samples)
                cher3_int = integrate_waveforms(tree[chrnkov3].array(library="np"), window, baseline_samples)

            # Masks
            muon_mask     = muon_int > muon_threshold
            positron_mask = (muon_int < muon_threshold) & (psd_int > psd_threshold)

            cher_data = [cher1_int, cher2_int, cher3_int]
            cher_labels = ["Cherenkov 1", "Cherenkov 2", "Cherenkov 3"]

            # Figure with 3 side-by-side subplots
            fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)

            for ax, ch, label in zip(axes, cher_data, cher_labels):
                bins = np.linspace(0, np.max(ch), 100)
                ax.hist(ch[muon_mask], bins=bins, alpha=0.6, label="Muons", histtype='step', color='blue')
                ax.hist(ch[positron_mask], bins=bins, alpha=0.6, label="Positrons", histtype='step', color='green')
                ax.hist(ch, bins=bins, histtype='step', color='black', linewidth=1.1, label="All events")
                ax.set_yscale("log")
                ax.set_xlabel("ADC")
                ax.set_title(label)
                ax.set_xlim(0, 30000)

                if ax == axes[0]:
                    ax.set_ylabel("Counts (log scale)")
                ax.legend(fontsize=8)

            plt.suptitle(f"Cherenkov Counters — Run {run_number}, {beam_energy}", fontsize=14)
            plt.tight_layout(rect=[0,0,1,0.95])

            output_file = os.path.join(output_folder, f"{particle_type}_Run{run_number}_energy_{beam_energy}.pdf")
            plt.savefig(output_file)
            plt.close()
            print(f"Saved PDF → {output_file}")

        except Exception as e:
            print(f"Failed to process {fname}: {e}")
