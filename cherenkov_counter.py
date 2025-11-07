# cherenkov_integrals_with_waveforms.py
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# Cherenkov branches
chrnkov1 = "DRS_Board7_Group2_Channel5"
chrnkov2 = "DRS_Board7_Group2_Channel6"
chrnkov3 = "DRS_Board7_Group2_Channel7"

basedir = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/"
outdir  = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Waveforms"

os.makedirs(outdir, exist_ok=True)


def integrate_waveforms(events, window=100, baseline_samples=20):
    """Compute baseline-subtracted integrals for each waveform."""
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)

        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)

        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)  # flip sign → positive area
    return np.array(integrals)


def plot_integrals(run_number, beam_energy, beam_type, ints1, ints2, ints3):
    """Make and save histograms for Cherenkov integrals (linear scale)."""
    plt.figure(figsize=(8,5))
    plt.hist(ints1, bins=200, histtype="step", label="Cherenkov 1")
    plt.hist(ints2, bins=200, histtype="step", label="Cherenkov 2")
    plt.hist(ints3, bins=200, histtype="step", label="Cherenkov 3")
    plt.xlabel("Integrated ADC (area)")
    plt.ylabel("Number of events")
    plt.title(f"Cherenkov Integrals\nRun {run_number}, {beam_energy} GeV {beam_type}")
    plt.legend()
    plt.yscale("linear")
    plt.xlim(0,40000)
    plt.tight_layout()

    out_file = os.path.join(
        outdir, f"Cherenkov_run{run_number}_{beam_energy}GeV_{beam_type}_linear.pdf"
    )
    plt.savefig(out_file)
    plt.close()
    print(f"Saved histogram → {out_file}")


def plot_waveforms(run_number, beam_energy, beam_type, data1, data2, data3, n_show=1000):
    """Overlay a few example waveforms for each Cherenkov counter."""
    plt.figure(figsize=(10,8))

    for i, data in enumerate([data1, data2, data3], start=1):
        plt.subplot(3, 1, i)
        num_to_plot = min(n_show, len(data))
        for w in data[:num_to_plot]:
            plt.plot(np.arange(len(w)), w, alpha=0.3, lw=0.8)
        plt.title(f"Cherenkov {i} Waveforms ({beam_energy} GeV {beam_type})")
        plt.xlabel("Sample index")
        plt.ylabel("ADC counts")

    plt.tight_layout()
    out_file = os.path.join(
        outdir, f"Waveforms_run{run_number}_{beam_energy}GeV_{beam_type}.pdf"
    )
    plt.savefig(out_file)
    plt.close()
    print(f"Saved waveform overlay → {out_file}")


def analyze_cherenkov(file_path, run_number, beam_energy, beam_type):
    """Load a ROOT file, compute integrals for 3 Cherenkov counters, and plot histograms."""
    with uproot.open(file_path) as f:
        tree = f["EventTree"]
        data1 = tree[chrnkov1].array(library="np")
        data2 = tree[chrnkov2].array(library="np")
        data3 = tree[chrnkov3].array(library="np")

    print(f"Loaded {len(data1)} events from {file_path}")

    # Plot waveform overlays
    plot_waveforms(run_number, beam_energy, beam_type, data1, data2, data3)

    # Compute integrals
    # ints1 = integrate_waveforms(data1)
    # ints2 = integrate_waveforms(data2)
    # ints3 = integrate_waveforms(data3)

    # Plot histograms of integrals
    #plot_integrals(run_number, beam_energy, beam_type, ints1, ints2, ints3)


# --- File dictionaries ---
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


def run_all():
    for fname, energy in positron_files.items():
        run_number = fname.split('_')[0].replace("run", "")
        analyze_cherenkov(os.path.join(basedir, fname), run_number, energy, "positron")

    for fname, energy in pion_files.items():
        run_number = fname.split('_')[0].replace("run", "")
        analyze_cherenkov(os.path.join(basedir, fname), run_number, energy, "pion")

    for fname, energy in muon_files.items():
        run_number = fname.split('_')[0].replace("run", "")
        analyze_cherenkov(os.path.join(basedir, fname), run_number, energy, "muon")


if __name__ == "__main__":
    run_all()
