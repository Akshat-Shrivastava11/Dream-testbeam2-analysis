import uproot
import matplotlib.pyplot as plt
import numpy as np
import os

# ================================
#   Branch names
# ================================
psd_branch   = "DRS_Board7_Group1_Channel1"
chrnkov1     = "DRS_Board7_Group2_Channel5"
chrnkov2     = "DRS_Board7_Group2_Channel6"
chrnkov3     = "DRS_Board7_Group2_Channel7"

basedir = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/"
outdir  = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD_histograms/"
os.makedirs(outdir, exist_ok=True)

# ================================
#   Integrate waveform (same as ref)
# ================================
def integrate_waveforms(events, window=100, baseline_samples=20):
    integrals = []
    for event in events:
        wf = np.array(event)
        baseline = np.mean(wf[:baseline_samples])
        corr = wf - baseline
        peak = np.argmin(corr)

        start = max(0, peak - window)
        end   = min(len(corr), peak + window)

        area = np.trapezoid(corr[start:end], dx=1)
        integrals.append(-area)    # positive area
    return np.array(integrals)

# ================================
#   Plot PSD histogram
# ================================
def make_psd_histogram(filepath, run_number, beam_energy, beam_type):

    with uproot.open(filepath) as f:
        tree = f["EventTree"]   # <-- FIXED

        psd_data  = tree[psd_branch].array(library="np")
        ch1_data  = tree[chrnkov1].array(library="np")
        ch2_data  = tree[chrnkov2].array(library="np")
        ch3_data  = tree[chrnkov3].array(library="np")

    # integrate all
    psd_int = integrate_waveforms(psd_data)
    ch1_int = integrate_waveforms(ch1_data)
    ch2_int = integrate_waveforms(ch2_data)
    ch3_int = integrate_waveforms(ch3_data)

    # Cherenkov condition
    mask = (ch1_int > 1000) | (ch2_int > 1000) | (ch3_int > 1000)
    psd_fire = psd_int[mask]

    # ========= Plot =========
    plt.figure(figsize=(8,6))
    plt.hist(psd_int, bins=120, alpha=0.40, label="All events")
    plt.hist(psd_fire, bins=120, alpha=0.55, label="Cherenkov > 1000")

    plt.xlabel("PSD Integral (baseline-subtracted)")
    plt.ylabel("Counts")
    plt.title(
        f"PSD Histogram (all cherenkov fire)\n"
        f"Run {run_number} • {beam_energy} GeV {beam_type}"
    )
    plt.legend()

    savepath = os.path.join(outdir, f"PSD_run{run_number}_{beam_energy}GeV_{beam_type}.pdf")
    plt.savefig(savepath)
    plt.close()
    print(f"✅ Saved PSD hist → {savepath}")


# ================================
#   File dictionary
# ================================
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

# ================================
#   Run all files
# ================================
for fname, energy in positron_files.items():
    run_number = fname.split("_")[0].replace("run", "")
    filepath = os.path.join(basedir, fname)
    make_psd_histogram(filepath, run_number, energy, "positron")
