import uproot
import matplotlib.pyplot as plt
import numpy as np

# PSD branch and file
PSD = "DRS_Board7_Group1_Channel1"
muoncounter = "DRS_Board7_Group2_Channel4"
#file_path = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/run1437_250927003120.root"
# run_number = '1437'
# position = '1'
# beam_energy = '60GeV'


file_path = "/lustre/research/hep/jdamgov/HG-DREAM/CERN/ROOT/run1412_250925174602.root"
run_number = '1412'
position = '1'
beam_energy = 'positron 60GeV'


# Number of waveforms to check
n_plot = 200  # can be larger to filter enough events
baseline_samples = 20  # number of samples to compute baseline
adc_threshold = 0      # min or max ADC must exceed ±3

# Load PSD events
with uproot.open(file_path) as f:
    tree = f["EventTree"]
    events = tree[PSD].array(library="np")

# Baseline-subtract and filter waveforms
filtered_events = []
count = 0
for event in events:
    event_np = np.array(event)
    baseline = np.mean(event_np[:baseline_samples])
    corrected = event_np - baseline
    
    if np.min(corrected) <= -adc_threshold or np.max(corrected) >= adc_threshold:
        filtered_events.append(corrected)
        count += 1

filtered_events = np.array(filtered_events)
print(f"Plotting {count} events out of {n_plot}")

# Plot filtered waveforms
plt.figure(figsize=(12,8))
for event in filtered_events:
    plt.plot(np.arange(len(event)), event, alpha=0.05, color="blue")

plt.xlabel("Time sample (index)")
plt.ylabel("ADC count (baseline subtracted)")
plt.title(f"PSD Waveforms (Baseline-Subtracted)\nRun {run_number}, Position #{position}, Energy {beam_energy}\nFiltered by ±{adc_threshold} ADC")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save as PDF
output_file = f"PSD_Waveforms_Filtered_BaselineSub_Run{run_number}_Pos{position}_{beam_energy}.pdf"
plt.savefig(output_file)
plt.close()
print(f"Saved waveform plot to {output_file}")

# Parameters
n_plot = 2000           # number of PSD events to check
baseline_samples = 30   # baseline samples for subtraction
adc_threshold = 40      # PSD selection threshold

# Load PSD and muon counter
with uproot.open(file_path) as f:
    tree = f["EventTree"]
    psd_events = tree[PSD].array(library="np")
    muon_events = tree[muoncounter].array(library="np")

# Select PSD events based on min/max ADC
selected_indices = []
for i, event in enumerate(psd_events[:n_plot]):
    event_np = np.array(event)
    baseline = np.mean(event_np[:baseline_samples])
    corrected = event_np - baseline
    if np.min(corrected) <= -adc_threshold or np.max(corrected) >= adc_threshold:
        selected_indices.append(i)

print(f"Selected {len(selected_indices)} PSD events out of {n_plot} for plotting muon waveforms")

# Plot corresponding muon counter waveforms
plt.figure(figsize=(12,8))
for idx in selected_indices:
    muon_event = np.array(muon_events[idx])
    baseline_mu = np.mean(muon_event[:baseline_samples])
    corrected_mu = muon_event - baseline_mu
    plt.plot(np.arange(len(corrected_mu)), corrected_mu, alpha=0.05, color="green")

plt.xlabel("Time sample (index)")
plt.ylabel("ADC count (baseline subtracted)")
plt.title(f"Muon Counter Waveforms{run_number}, Position #{position}, Energy {beam_energy}")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save as PDF
output_file = f"MuonCounter_Waveforms_PSD_Run{run_number}_Pos{position}_{beam_energy}.pdf"
plt.savefig(output_file)
plt.close()
print(f"Saved waveform plot to {output_file}")