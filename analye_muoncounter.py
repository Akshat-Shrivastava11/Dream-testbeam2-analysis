import uproot
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np


#position #1 run1355 beam energy GeV
run_number = '1355'
postion = '#1'
beam_energy = '80GeV'

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

# === Example usage on one file/branch ===
file_path = "run1355_250924165834_converted.root"

with uproot.open(file_path) as f:
    tree = f["EventTree"]
    branch = "DRS_Board7_Group2_Channel4"
    
    events = tree[branch].array(library="np")  # ~10k events
    
    min_values, max_values = pulse_extractor(events)
    print("Events:", len(events))
    print("First 10 min values:", min_values[:10])
    print("First 10 max values:", max_values[:10])
    
    # Plot histograms of min and max
    plt.figure(figsize=(10,6))
    plt.hist(min_values, bins=500, histtype="step", color="red", label="Min values")
    plt.hist(max_values, bins=100, histtype="step", color="blue", label="Max values")
    plt.xlabel("ADC counts (baseline subtracted)")
    plt.xlim(-100,100)
    plt.ylabel("Events")
    plt.title(f"Pulse Min/Max Distribution Run: {run_number} position: {postion} beam energy: {beam_energy}({len(events)} events)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pulse_MinMax_Distribution.pdf")
    print("Saved Pulse_MinMax_Distribution.pdf")

# Inspect keys and branches
# with uproot.open(file_path) as f:
#     print("Keys in file:", f.keys())
#     tree = f["EventTree"]

#     print("\nBranches in EventTree:")
#     for branch in tree.keys():
#         print(branch)






'''
# Analyze and plot muon counter data
with uproot.open(file_path) as f:
    tree = f["EventTree"]
    muon_counter_branch_new = "DRS_Board7_Group2_Channel4"
    timestampbegin = "timestampbegin"
    timestampend = "timestampend"

   #muon_counter_branch_old = "DRS_Board7_Gr its off
    if muon_counter_branch_new in tree.keys():
        data = tree[muon_counter_branch_new].array(library="np")
        timein = tree[timestampbegin].array(library="np")
        timeout = tree[timestampend].array(library="np")
        #if data.dtype == object:  # jagged case
        #    data = np.concatenate(data)
        # else:  # already flat numeric array
        #    data = data

        print(f"First 20 entries of {muon_counter_branch_new}: {data[:20]}")
        print(f" Time begin {tree[timestampbegin].array(library='np')[:20]}")
        print(f" Time end {tree[timestampend].array(library='np')[:20]}")
        plt.figure(figsize=(12,8))
        plt.hist(data,bins=200, histtype='step', label=muon_counter_branch_new)
        plt.xlabel("Muon Counter Value")
        plt.xlim(0,2500)
        plt.ylabel("ADC")
        plt.yscale("log")
        plt.title(f"Histogram of Muon Counter Values run {run_number} postion {postion} beam energy {beam_energy}")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("New_Muon_Counter_flattened.pdf")
        print("Saved plot as New_Muon_Counter_flattened.pd")
    

# # Define PID branches
# drs_pids = [f"DRS_Board{i}_PID" for i in range(8)]      # DRS boards 0–7
# fers_pids = [f"FERS_Board{i}_PID" for i in range(16)]   # FERS boards 0–15

# with uproot.open(file_path) as f:
#       tree = f["EventTree"]
#       branches_to_study = ['run_n','event_n','event_flag','device_n','trigger_n','timestampbegin','timestampend','DRS_Board7_Group2_Channel4']
#       for branch in branches_to_study:
#           if branch in tree.keys():
#               data = tree[branch].array(library="np")
#               print(f"First 20 entries of {branch}: {data[:20]}")  

#     # --- DRS Plot ---
#     plt.figure(figsize=(10,6))
#     for branch in drs_pids:
#         if branch in tree.keys():
#             data = tree[branch].array(library="np")
#             print(f"First 20 entries of {branch}: {data[:20]}")
#             plt.hist(data, bins=100, histtype='step', label=branch)

#     plt.xlabel("PID value")
#     plt.ylabel("Entries")
#     plt.title("Overlayed Histogram of DRS Board PIDs")
#     plt.legend(fontsize=8, ncol=2)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("Overlayed_DRS_Board_PIDs.pdf")
#     print("Saved plot as Overlayed_DRS_Board_PIDs.pdf")

#     # --- FERS Plot ---
#     plt.figure(figsize=(10,6))
#     for branch in fers_pids:
#         if branch in tree.keys():
#             data = tree[branch].array(library="np")
#             print(f"First 20 entries of {branch}: {data[:20]}")
#             plt.hist(data, bins=100, histtype='step', label=branch)

#     plt.xlabel("PID value")
#     plt.ylabel("Entries")
#     plt.title("Overlayed Histogram of FERS Board PIDs")
#     plt.legend(fontsize=8, ncol=3)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("Overlayed_FERS_Board_PIDs.pdf")
#     print("Saved plot as Overlayed_FERS_Board_PIDs.pdf")
'''