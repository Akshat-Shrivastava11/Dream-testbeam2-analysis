import uproot
import matplotlib.pyplot as plt
import numpy as np

# file_path = "data_testbeam/testbeam_round2/run1355_250924165834_converted.root"
# muoncounter = "DRS_Board7_Group2_Channel4"
# run_number = '1355'
# position = '#1'
# beam_energy = '80GeV positrons'

# file_path = '/lustre/work/akshriva/Dream/testbeam02/data_testbeam/testbeam_round2/run1410_250925145231.root'
# muoncounter = "DRS_Board7_Group2_Channel4"
# run_number = '1410'
# position = '#3'
# beam_energy = '80GeV positrons'

# file_path = 'data_testbeam/testbeam_round2/run1431_250926195104.root'
# muoncounter = "DRS_Board7_Group2_Channel4"
# run_number = '1431'
# position = '#1'
# beam_energy = '100 GeV pions+'

file_path = 'data_testbeam/testbeam_round2/run1420_250926093800.root'
muoncounter = "DRS_Board7_Group2_Channel4"
run_number = '1420'
position = '#1'
beam_energy = '110 GeV muons+'

with uproot.open(file_path) as f:
    tree = f["EventTree"]
    events = tree[muoncounter].array(library="np")  # object array

print(f"Loaded {len(events)} events.")

import numpy as np

def integrate_waveforms(events, window=100, baseline_samples=20):
    """
    Compute baseline-corrected integrals for each waveform in events.

    Parameters
    ----------
    events : list of arrays
        Each element is an event (array of waveform samples).
    window : int, optional
        Number of samples on each side of the peak to integrate (default: 100).
    baseline_samples : int, optional
        Number of samples at the start used for baseline calculation (default: 20).

    Returns
    -------
    integrals : list of floats
        List of integrated (negative) areas for each waveform.
    """
    integrals = []
    for event in events:
        event_np = np.array(event)
        baseline = np.mean(event_np[:baseline_samples])
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)

        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)

        #area = np.trapz(corrected[start:end], dx=1)
        area = np.trapezoid(corrected[start:end], dx=1)
        integrals.append(-area)

    return integrals


# Compute integral (area) for each waveform
integrals = []
window = 100 # samples around the minimum to integrate
for event in events:
    #print(f'Processing event with {len(event)} samples')
    event_np = np.array(event)
    baseline = np.mean(event_np[:20])      # average of first 20 samples
    corrected = event_np - baseline        # baseline subtraction
    peak_index = np.argmin(corrected)
    
    # Define integration window
    start = max(0, peak_index - window)
    end   = min(len(corrected), peak_index + window)
    
    # Integrate only in that window
    area = np.trapz(corrected[start:end], dx=1)
    integrals.append(-area)


integrals = np.array(integrals)
print(f"First 10 integrals: {integrals[:10]}")

# Optional: plot histogram of waveform integrals
noise  = integrals[integrals <= 5000]
muons = integrals[integrals > 5000]
muon_contamination = len(muons) / len(integrals)
plt.figure(figsize=(8,5))

# Plot low integrals (gray)
plt.hist(noise, bins=1000, histtype="stepfilled", 
         color='gray', alpha=0.5, label="Noise")

# Plot high integrals (blue, overlaid)
plt.hist(muons, bins=1000, histtype="stepfilled", 
         color='blue', alpha=0.7, label=f"Muons , contamination = {muon_contamination:.2%}")

plt.xlabel("Integrated ADC (area)")
plt.ylabel("Number of events")
plt.title(f"Muon counter waveform integrals\nRun {run_number}, {beam_energy}")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(f"Muoncounts_for_{run_number}_energy_{beam_energy}.pdf")
plt.close()

print(f"Saved histogram of integrals to Muoncounts_for_{run_number}_energy_{beam_energy}.pdf")