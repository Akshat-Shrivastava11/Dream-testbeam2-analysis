import uproot
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Output directory
# ============================================================
outdir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/beam_fraction_plots"
os.makedirs(outdir, exist_ok=True)

# ============================================================
# File lists
# ============================================================
positron_files = {
    'run1527_250929001555.root': '100',
    'run1411_250925154340.root': '120',
    'run1422_250926102502.root': '110',
    'run1409_250925135843.root': '80',
    'run1526_250928235028.root': '60',
    'run1525_250928232144.root': '40',
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

basedir = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT"

# ============================================================
# Channels
# ============================================================
psd_branch   = "DRS_Board7_Group1_Channel1"
muon_branch  = "DRS_Board7_Group2_Channel4"
ck1_branch   = "DRS_Board7_Group2_Channel5"
ck2_branch   = "DRS_Board7_Group2_Channel6"
ck3_branch   = "DRS_Board7_Group2_Channel7"

# ============================================================
# Analysis thresholds
# ============================================================
window           = 100
baseline_samples = 20
muon_threshold   = 5000
cherenkov_cut    = 1000

# ============================================================
# Waveform integration (batched)
# NOTE: this function expects an iterable of events and returns a numpy array
# ============================================================
def integrate_waveform(events, window=100, baseline_samples=20):
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

# ============================================================
# Cherenkov firing lookup
# ============================================================
CHERENKOV_TABLE = {
    "positron": {
        5:(1,1,1),10:(1,1,1),20:(1,1,1),30:(1,1,1),40:(1,1,1),
        60:(1,1,1),80:(1,1,1),100:(1,1,1),110:(1,1,1),120:(1,1,1),160:(1,1,1),170:(1,1,1),
    },
    "pion": {
        5:(0,0,0),10:(0,0,0),20:(0,0,1),30:(0,1,1),40:(1,1,1),
        60:(1,1,1),80:(1,1,1),100:(1,1,1),110:(1,1,1),120:(1,1,1),160:(1,1,1),170:(1,1,1),
    },
    "kaon": {
        5:(0,0,0),10:(0,0,0),20:(0,0,0),30:(0,0,0),40:(0,0,0),
        60:(0,0,0),80:(0,1,1),100:(0,1,1),110:(0,1,1),120:(1,1,1),160:(1,1,1),170:(1,1,1),
    },
    "proton": {
        5:(0,0,0),10:(0,0,0),20:(0,0,0),30:(0,0,0),40:(0,0,0),
        60:(0,0,0),80:(0,0,0),100:(0,0,0),110:(0,0,0),120:(0,0,1),160:(0,1,1),170:(0,1,1),
    }
}

# ============================================================
# PID classifier
# ============================================================
def classify_events(psd, mu, ck1, ck2, ck3, energy):

    e = int(energy)
    print(f"  Classifying events for E={e} GeV")

    ck_pos = CHERENKOV_TABLE["positron"].get(e)
    ck_pi  = CHERENKOV_TABLE["pion"].get(e)
    ck_k   = CHERENKOV_TABLE["kaon"].get(e)
    ck_p   = CHERENKOV_TABLE["proton"].get(e)

    if ck_pos is None or ck_pi is None or ck_k is None or ck_p is None:
        raise ValueError(f"No CHERENKOV_TABLE entries for energy {e}")

    print(f"    CK pattern positron={ck_pos}, pion={ck_pi}, kaon={ck_k}, proton={ck_p}")

    def match(exp, val):
        # val is array-like
        return (val > cherenkov_cut) if exp else (val <= cherenkov_cut)

    def ck_match(pattern):
        m1 = match(pattern[0], ck1)
        m2 = match(pattern[1], ck2)
        m3 = match(pattern[2], ck3)
        # optionally print summary (counts) instead of full arrays
        print(f"    CK match pattern {pattern} → counts ({np.sum(m1)},{np.sum(m2)},{np.sum(m3)})")
        return m1 & m2 & m3

    is_muon = (mu > muon_threshold)

    is_positron = (
        (psd > 5000) & (ck_match(ck_pos)) &
        (mu < muon_threshold)
    )

    is_hadron = (psd < 5000) & (mu < muon_threshold)

    is_pion   = is_hadron & ck_match(ck_pi)
    is_kaon   = is_hadron & ck_match(ck_k)
    is_proton = is_hadron & ck_match(ck_p)

    print(f"    Muons={np.sum(is_muon)}, e⁺={np.sum(is_positron)}, π={np.sum(is_pion)}, K={np.sum(is_kaon)}, p={np.sum(is_proton)}")

    return is_muon, is_positron, is_pion, is_kaon, is_proton

# ============================================================
# Process a ROOT file
# ============================================================
def process_file(filepath, energy, beam_type):

    print(f"\n===================================================")
    print(f"Processing file {filepath}")
    print(f"Beam={beam_type}, Energy={energy} GeV")
    print(f"===================================================")

    f = uproot.open(filepath)
    tree = f["EventTree"]

    print("Loading branches...")

    # load arrays
    psd_list  = tree[psd_branch].array(library="np")
    mu_list   = tree[muon_branch].array(library="np")
    ck1_list  = tree[ck1_branch].array(library="np")
    ck2_list  = tree[ck2_branch].array(library="np")
    ck3_list  = tree[ck3_branch].array(library="np")

    print("Integrating all waveforms (batched integrator)...")

    # use the batched integrator you provided (returns arrays)
    psd_int = integrate_waveform(psd_list, window, baseline_samples)
    print(f"  PSD integrals: n={len(psd_int)}, median={np.median(psd_int):.1f}, p90={np.percentile(psd_int,90):.1f}")
    mu_int  = integrate_waveform(mu_list, window, baseline_samples)
    ck1_int = integrate_waveform(ck1_list, window, baseline_samples)
    ck2_int = integrate_waveform(ck2_list, window, baseline_samples)
    ck3_int = integrate_waveform(ck3_list, window, baseline_samples)

    print("Running classifier...")

    is_mu, is_e, is_pi, is_k, is_p = classify_events(
        psd_int, mu_int, ck1_int, ck2_int, ck3_int, energy
    )

    N = len(psd_int)
    print(f"Total events: {N}")

    fracs = {
        "energy": int(energy),
        "muon": np.sum(is_mu) / N,
        "positron": np.sum(is_e) / N,
        "pion": np.sum(is_pi) / N,
        "kaon": np.sum(is_k) / N,
        "proton": np.sum(is_p) / N,
    }

    print("  Fractions:")
    for k,v in fracs.items():
        if k != "energy":
            print(f"    {k}: {v:.4f}")

    return fracs

# ============================================================
# Run over beam type
# ============================================================
def run_beam(beam_type, filedict):

    print(f"\n\n#############################################")
    print(f" Running beam type: {beam_type}")
    print(f"#############################################\n")

    results = []

    for fname, E in filedict.items():
        full = os.path.join(basedir, fname)
        print(f"\n--> Starting file: {fname}")
        results.append(process_file(full, E, beam_type))

    results = sorted(results, key=lambda x: x["energy"])

    energies = [r["energy"] for r in results]
    mu_f     = [r["muon"] for r in results]
    e_f      = [r["positron"] for r in results]
    pi_f     = [r["pion"] for r in results]
    k_f      = [r["kaon"] for r in results]
    p_f      = [r["proton"] for r in results]

    print("\nPlotting...")

    plt.figure(figsize=(8,6))
    plt.plot(energies, mu_f, 'o-', label="Muon fraction")
    plt.plot(energies, e_f,  'o-', label="Positron fraction")
    plt.plot(energies, pi_f, 'o-', label="Pion fraction")
    plt.plot(energies, k_f,  'o-', label="Kaon fraction")
    plt.plot(energies, p_f,  'o-', label="Proton fraction")

    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel("Fraction")
    plt.title(f"PID Fractions vs Energy — {beam_type}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    pdfname = os.path.join(outdir, f"beam_fractions_{beam_type}.pdf")
    plt.savefig(pdfname)
    plt.close()

    print(f"Saved plot → {pdfname}\n")

# ============================================================
# Run everything
# ============================================================
run_beam("positron", positron_files)
run_beam("pion", pion_files)
run_beam("muon", muon_files)

print("\n\nAll done!")
