import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
import csv

# ----------------------------
# User settings
# ----------------------------
basedir = '/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/'

# Beam file dictionaries (use your existing lists)
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

all_files = {"positron": positron_files, "pion": pion_files, "muon": muon_files}

# Channels
psd         = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"
chrnkov1    = "DRS_Board7_Group2_Channel5"
chrnkov2    = "DRS_Board7_Group2_Channel6"
chrnkov3    = "DRS_Board7_Group2_Channel7"

# thresholds & integration
muon_threshold = 5000
psd_threshold  = 2000
window = 50
baseline_samples = 20

# purity target for suggesting thresholds
target_purity = 0.95

# outputs
output_folder = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_dynamic_cuts"
os.makedirs(output_folder, exist_ok=True)
summary_csv = os.path.join(output_folder, "dynamic_cuts_summary.csv")

# ----------------------------
# helper: waveform integrate
# ----------------------------
def integrate_waveforms(events, window=50, baseline_samples=20):
    integrals = []
    for event in events:
        event_np = np.array(event)
        if event_np.size == 0:
            integrals.append(0.0)
            continue
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)
        start = max(0, peak_index - window)
        end = min(len(corrected), peak_index + window)
        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)
    return np.array(integrals)

# ----------------------------
# process a single run: compute suggestions and plots
# ----------------------------
def process_run(file_path, run_label, out_dir, target_purity=0.95):
    try:
        with uproot.open(file_path) as f:
            tree = f["EventTree"]
            psd_arr   = integrate_waveforms(tree[psd].array(library="np"), window, baseline_samples)
            muon_arr  = integrate_waveforms(tree[muoncounter].array(library="np"), window, baseline_samples)
            cher1_arr = integrate_waveforms(tree[chrnkov1].array(library="np"), window, baseline_samples)
            cher2_arr = integrate_waveforms(tree[chrnkov2].array(library="np"), window, baseline_samples)
            cher3_arr = integrate_waveforms(tree[chrnkov3].array(library="np"), window, baseline_samples)
    except Exception as e:
        print(f"ERROR opening {file_path}: {e}")
        return None

    # define reference pion (PSD+muon)
    pion_ref = (muon_arr < muon_threshold) & (psd_arr < psd_threshold)
    n_pion_ref = int(np.sum(pion_ref))
    total_ev = len(muon_arr)
    if total_ev == 0:
        print(f"Empty tree in {file_path}")
        return None

    # arrays
    chers = [cher1_arr, cher2_arr, cher3_arr]
    cher_names = ["Cher1", "Cher2", "Cher3"]

    # per-counter scan
    per_counter_choice = {}
    purity_eff_curves = []
    for i, ch in enumerate(chers):
        # avoid extreme outliers: scan to 99.5 percentile
        max_scan = np.percentile(ch, 99.5)
        threshs = np.linspace(0, max_scan, 160)
        purities = np.zeros_like(threshs)
        effs = np.zeros_like(threshs)
        for j, T in enumerate(threshs):
            sel = (muon_arr < muon_threshold) & (psd_arr < psd_threshold) & (ch < T)
            nsel = np.sum(sel)
            if nsel == 0:
                purities[j] = 0.0
                effs[j] = 0.0
            else:
                purities[j] = np.sum(sel & pion_ref) / nsel
                effs[j] = np.sum(sel & pion_ref) / n_pion_ref if n_pion_ref > 0 else 0.0

        # find first threshold achieving target purity; else fallback to median
        idx = np.where(purities >= target_purity)[0]
        if len(idx) > 0:
            chosen_T = float(threshs[idx[0]])
            chosen_purity = float(purities[idx[0]])
            chosen_eff = float(effs[idx[0]])
        else:
            # if never reach target, pick threshold at max purity
            idx_max = int(np.argmax(purities))
            chosen_T = float(threshs[idx_max])
            chosen_purity = float(purities[idx_max])
            chosen_eff = float(effs[idx_max])

        per_counter_choice[cher_names[i]] = {
            "threshold": chosen_T,
            "purity": chosen_purity,
            "efficiency": chosen_eff
        }
        purity_eff_curves.append((threshs, purities, effs))

    # combined strategies
    # 1) require all three < their chosen thresholds (AND)
    thresholds = [per_counter_choice[c]["threshold"] for c in cher_names]
    all_sel = (muon_arr < muon_threshold) & (psd_arr < psd_threshold) & \
              (cher1_arr < thresholds[0]) & (cher2_arr < thresholds[1]) & (cher3_arr < thresholds[2])
    n_all = np.sum(all_sel)
    purity_all = (np.sum(all_sel & pion_ref) / n_all) if n_all>0 else 0.0
    eff_all = (np.sum(all_sel & pion_ref) / n_pion_ref) if n_pion_ref>0 else 0.0

    # 2) require any one < threshold (OR) — more efficient, less pure
    any_sel = (muon_arr < muon_threshold) & (psd_arr < psd_threshold) & \
              ((cher1_arr < thresholds[0]) | (cher2_arr < thresholds[1]) | (cher3_arr < thresholds[2]))
    n_any = np.sum(any_sel)
    purity_any = (np.sum(any_sel & pion_ref) / n_any) if n_any>0 else 0.0
    eff_any = (np.sum(any_sel & pion_ref) / n_pion_ref) if n_pion_ref>0 else 0.0

    # 3) require majority (>=2 of 3) below thresholds
    conds = [(cher1_arr < thresholds[0]), (cher2_arr < thresholds[1]), (cher3_arr < thresholds[2])]
    majority_sel = (muon_arr < muon_threshold) & (psd_arr < psd_threshold) & \
                   ( (conds[0].astype(int) + conds[1].astype(int) + conds[2].astype(int)) >= 2 )
    n_maj = np.sum(majority_sel)
    purity_maj = (np.sum(majority_sel & pion_ref) / n_maj) if n_maj>0 else 0.0
    eff_maj = (np.sum(majority_sel & pion_ref) / n_pion_ref) if n_pion_ref>0 else 0.0

    # Prepare summary dict
    summary = {
        "run": run_label,
        "n_total": int(total_ev),
        "n_pion_ref": int(n_pion_ref),
        "per_counter": per_counter_choice,
        "AND_n": int(n_all), "AND_purity": float(purity_all), "AND_eff": float(eff_all),
        "OR_n": int(n_any), "OR_purity": float(purity_any), "OR_eff": float(eff_any),
        "MAJ_n": int(n_maj), "MAJ_purity": float(purity_maj), "MAJ_eff": float(eff_maj)
    }

    # ----- plotting: 1) Cherenkov histograms (mu/pos/pion_ref/all)  2) purity/eff curves -----
    fig, axs = plt.subplots(2, 3, figsize=(18,9), gridspec_kw={'height_ratios':[1,1]})
    # top row: histograms (mu, pos, pion_ref, all)
    masks = {
        "Muons": (muon_arr > muon_threshold),
        "Positrons": (muon_arr < muon_threshold) & (psd_arr > psd_threshold),
        "Pion_ref": pion_ref,
        "All": np.ones_like(muon_arr, dtype=bool)
    }
    colors = {"Muons":"C0", "Positrons":"C2", "Pion_ref":"C1", "All":"k"}
    for i, ch in enumerate(chers):
        ax = axs[0, i]
        maxx = np.percentile(ch, 99.5)
        bins = np.linspace(0, maxx, 120)
        for lab, m in masks.items():
            ax.hist(ch[m], bins=bins, histtype='step', label=f"{lab} ({np.sum(m)})", color=colors[lab], alpha=0.8)
        ax.set_yscale('log')
        ax.set_xlabel(f"{cher_names[i]} ADC")
        if i==0:
            ax.set_ylabel("counts (log)")
        ax.set_title(f"{cher_names[i]} hist")
        ax.legend(fontsize=8)

    # bottom row: purity & eff curves
    for i, (threshs, purities, effs) in enumerate(purity_eff_curves):
        ax = axs[1, i]
        ax.plot(threshs, purities, label='purity')
        ax.plot(threshs, effs, label='efficiency')
        chosen_T = per_counter_choice[cher_names[i]]['threshold']
        ax.axvline(chosen_T, color='k', linestyle='--', label=f"chosen T={int(chosen_T)}")
        ax.set_xlabel(f"{cher_names[i]} threshold")
        ax.set_ylim(0,1.02)
        if i==0:
            ax.set_ylabel("fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle(f"Dynamic Cherenkov cuts — {run_label}")
    plt.tight_layout(rect=[0,0,1,0.96])

    out_pdf = os.path.join(out_dir, f"{run_label}_dynamic_cuts.pdf")
    plt.savefig(out_pdf)
    plt.close()

    # return summary
    return summary

# ----------------------------
# loop all files and write CSV summary
# ----------------------------
fieldnames = [
    "run","n_total","n_pion_ref",
    "Cher1_thr","Cher1_purity","Cher1_eff",
    "Cher2_thr","Cher2_purity","Cher2_eff",
    "Cher3_thr","Cher3_purity","Cher3_eff",
    "AND_n","AND_purity","AND_eff",
    "OR_n","OR_purity","OR_eff",
    "MAJ_n","MAJ_purity","MAJ_eff"
]

with open(summary_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for particle_type, files_dict in all_files.items():
        for fname, energy in files_dict.items():
            file_path = os.path.join(basedir, fname)
            run_label = f"{particle_type}_Run{fname.split('_')[0].replace('run','')}_E{energy}GeV"
            print(f"Processing {run_label} ...")
            summary = process_run(file_path, run_label, output_folder, target_purity=target_purity)
            if summary is None:
                print(f"  skipped {run_label}")
                continue

            row = {
                "run": summary["run"],
                "n_total": summary["n_total"],
                "n_pion_ref": summary["n_pion_ref"],
                "Cher1_thr": int(summary["per_counter"]["Cher1"]["threshold"]),
                "Cher1_purity": round(summary["per_counter"]["Cher1"]["purity"],4),
                "Cher1_eff": round(summary["per_counter"]["Cher1"]["efficiency"],4),
                "Cher2_thr": int(summary["per_counter"]["Cher2"]["threshold"]),
                "Cher2_purity": round(summary["per_counter"]["Cher2"]["purity"],4),
                "Cher2_eff": round(summary["per_counter"]["Cher2"]["efficiency"],4),
                "Cher3_thr": int(summary["per_counter"]["Cher3"]["threshold"]),
                "Cher3_purity": round(summary["per_counter"]["Cher3"]["purity"],4),
                "Cher3_eff": round(summary["per_counter"]["Cher3"]["efficiency"],4),
                "AND_n": summary["AND_n"], "AND_purity": round(summary["AND_purity"],4), "AND_eff": round(summary["AND_eff"],4),
                "OR_n": summary["OR_n"], "OR_purity": round(summary["OR_purity"],4), "OR_eff": round(summary["OR_eff"],4),
                "MAJ_n": summary["MAJ_n"], "MAJ_purity": round(summary["MAJ_purity"],4), "MAJ_eff": round(summary["MAJ_eff"],4),
            }
            writer.writerow(row)
            print(f"  done; wrote PDF + summary row for {run_label}")

print("All done. Summary CSV:", summary_csv)
print("PDFs in:", output_folder)
