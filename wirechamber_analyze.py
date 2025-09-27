import uproot
import matplotlib.pyplot as plt
import numpy as np


run_number = '1355'
position = '#1'
beam_energy = '80GeV'


def get_hit_times(events):
    """
    Given an array of events (each event = waveform samples),
    baseline subtract (mean of first 20 samples),
    return index of min (hit time) for each event.
    """
    hit_times = []
    
    for event in events:
        baseline = np.mean(event[:20])
        corrected = event - baseline
        hit_index = np.argmin(corrected)  # index of min after baseline subtraction
        hit_times.append(hit_index)
    
    return np.array(hit_times)


file_path = "run1355_250924165834_converted.root"

with uproot.open(file_path) as f:
    tree = f["EventTree"]

    # wire chamber channels
    L1_events = tree['DRS_Board7_Group0_Channel0'].array(library="np")
    R1_events = tree['DRS_Board7_Group0_Channel1'].array(library="np")
    U1_events = tree['DRS_Board7_Group0_Channel2'].array(library="np")
    D1_events = tree['DRS_Board7_Group0_Channel3'].array(library="np")


    #MCP channel for reference
    #do MCP/events

    # Get hit times (indices of minima)
    L1_times = get_hit_times(L1_events)
    R1_times = get_hit_times(R1_events)
    U1_times = get_hit_times(U1_events)
    D1_times = get_hit_times(D1_events)

    # X and Y position proxies
    x_positions = L1_times - R1_times
    y_positions = U1_times - D1_times

    print(f"Total events processed: {len(x_positions)}")
    print(f"X position range: {np.min(x_positions)} to {np.max(x_positions)}")
    print(f"Y position range: {np.min(y_positions)} to {np.max(y_positions)}")

    # --- 2D histogram ---
    plt.figure(figsize=(12,8))
    h = plt.hist2d(x_positions, y_positions, bins=100, cmap="viridis")
    plt.xlim(-250,250)
    plt.ylim(-250,250)
    plt.colorbar(h[3], label="Counts")
    plt.xlabel("X position (L1_time - R1_time)")
    plt.ylabel("Y position (U1_time - D1_time)")
    plt.title(f"WireChamber 2D Position - Run {run_number}, {beam_energy}, pos {position}")
    plt.savefig(f"WireChamber_2D_run{run_number}_{position}_{beam_energy}.pdf", dpi=300)

