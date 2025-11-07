import os
from PyPDF2 import PdfMerger
import re

dirs = [
    # "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Waveforms",
    # "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/muon_vs_cherenkov",
    # "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_vs_PSD_plots_2D_linear",
    # "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/PSD/stacked_plots_electroncontamination_muon_and_CherenkovCuts",
    #'/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_mu_e_plots'
    '/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_mu_e_post_tightpsd',
    '/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Cherenkov_Counter/Cherenkov_energywise_overlay'
    
]

for d in dirs:
    merger = PdfMerger()
    pdfs = sorted(
        [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith(".pdf")
        ],
        key=lambda x: int(re.search(r"run(\d+)", x).group(1)) if re.search(r"run(\d+)", x) else float("inf")
    )

    if not pdfs:
        print(f"⚠️ No PDFs found in {d}")
        continue

    for pdf in pdfs:
        merger.append(pdf)

    output_path = os.path.join(d, "combined_all_plots.pdf")
    merger.write(output_path)
    merger.close()

    print(f"✅ Combined {len(pdfs)} PDFs → {output_path}")
