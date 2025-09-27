#saturationstudy
import uproot
import matplotlib.pyplot as plt
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
        print(f'here is the avereage for event num {event}: {baseline}')
        corrected = event_np - baseline
        peak_index = np.argmin(corrected)

        start = max(0, peak_index - window)
        end   = min(len(corrected), peak_index + window)

        area = np.trapz(corrected[start:end], dx=1)
        integrals.append(-area)

    return integrals

def pulse_extractor(events):
    """
    Given an array of events (each event = waveform samples),
    subtract baseline (mean of first 20 samples), then return
    per-event min and max after baseline correction.
    """
    min_values = []
    max_values = []
    
    for event in events:
        baseline = np.mean(event[:20])       # average of first 20 entries
        print(f'here is the avereage for event num {event}: {baseline}')
        corrected = event - baseline         # baseline subtraction
        print(f'here is the min for event num {event}: {np.min(corrected)}')
        print(f'here is the max for event num {event}: {np.max(corrected)}')
        
        min_values.append(np.min(corrected))
        max_values.append(np.max(corrected))
    
    return np.array(min_values), np.array(max_values)

file_path = "data_testbeam/testbeam_round2/run1355_250924165834_converted.root"
channel = "DRS_Board2_Group2_Channel5"
run_number = '1355'
position = '#1'
beam_energy = '80GeV'

#integrate_waveforms
with uproot.open(file_path) as f:
    tree = f["EventTree"]
    events = tree[channel].array(library="np")  # object array
    print(f"Loaded {len(events)} events.")
    integrals = integrate_waveforms(events, window=100, baseline_samples=20)
    print(f"Computed integrals for {len(integrals)} events.")
    print("First 10 integrals:", integrals[:10])
    # Plot histogram of integrals
    plt.figure(figsize=(12,8))          
    plt.hist(integrals, bins=200, histtype="step", color="blue")
    plt.xlabel("Integrated ADC counts (baseline subtracted)")
    plt.ylabel("Events")
    plt.title(f"Waveform Integrals Run: {run_number} position: {position} beam energy: {beam_energy}({len(events)} events)")
    #plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f"Waveform_Integrals_for_{run_number}_{position}_{beam_energy}.pdf")
    print(f"Saved Waveform_Integrals_for_{run_number}_{position}_{beam_energy}.pdf")



#plot waveforms 
plt.figure(figsize=(12,8))
events = tree[channel].array(library="np")  # object array
for event in events:
    event_np = np.array(event)
    # Normalize to baseline
    plt.plot(np.arange(len(event_np)), event_np, alpha=0.05, color="blue")

plt.xlabel("Time sample (index)")
plt.ylabel("ADC count")
plt.title(" waveforms\nRun 1355, Position #1, 80GeV")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save as PDF
output_file = "Staturated_Waveforms_Run1355_Position1_80GeV.pdf"
plt.savefig(output_file)
plt.close()
print(f"Saved waveform plot to {output_file}")

# with uproot.open(file_path) as f:
#     print("Keys in file:", f.keys())

#     # Open the EventTree
#     tree = f["EventTree"]

#     print("\nBranches in EventTree:")
#     for branch in tree.keys():
#         print(branch)
