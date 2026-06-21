#!/usr/bin/env python3
"""
Helium threshold-Cherenkov PID study with one and two counters.

Counter 1 (C1)
--------------
Uses the supplied helium set-pressure values from the beam-plan table.  Those
values track below the pion threshold curve, so C1 is primarily an electron
counter: electrons fire, while pions, kaons, and protons remain OFF.

Counter 2 (C2)
--------------
Follows the proton threshold curve in exactly the same relative way that C1
follows the pion threshold curve.  At each beam energy:

    f(E)  = P_C1,supplied(E) / P_pi,table(E)
    P_C2(E) = f(E) * P_p,threshold(E)

This places C2 below the proton threshold but far above the pion threshold.
Therefore, for the nominal pressure schedules used here:

    e+      -> C1C2 = 11
    pi+     -> C1C2 = 01
    K+      -> C1C2 = 01
    proton  -> C1C2 = 00

The two counters consequently separate pions from protons.  They do not
separate pions from kaons in this particular configuration.

Outputs
-------
  * helium_one_and_two_counter_pi_proton_heatmaps.pdf
      Left:  one supplied counter (C1) only.
      Right: combined two-counter signatures (C1,C2).
  * helium_pi_proton_counter_pressure_schedule.pdf
      Pion / kaon / proton threshold curves plus C1 and proton-following C2.
  * helium_pi_proton_counter_pid_tables.xlsx
      Pressure schedule, all robust firing states, and heatmap tables.
  * helium_pi_proton_counter_pid_summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# ============================================================
# Output directory
# ============================================================

SAVE_DIR = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/TB2026/Calculations2"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Helium Cherenkov model
# ============================================================

MASSES_GEV = {
    "e+": 0.000511,
    "pi+": 0.13957039,
    "K+": 0.493677,
    "proton": 0.938272,
}

# n_He(P) = 1 + HELIUM_N_MINUS_1_AT_1_BAR * P[bar]
HELIUM_N_MINUS_1_AT_1_BAR = 3.553e-5

# Pressure uncertainty requested for the operating curves / PID robustness.
PRESSURE_RELATIVE_UNCERTAINTY = 0.10  # +/-10%


# ============================================================
# Physics helpers
# ============================================================

def beta_relativistic(total_energy_gev, mass_gev):
    """Relativistic beta = v/c for total energy E and rest mass m."""
    if total_energy_gev <= mass_gev:
        return 0.0
    return np.sqrt(1.0 - (mass_gev / total_energy_gev) ** 2)


def helium_refractive_index(pressure_bar):
    """Helium refractive index at pressure in bar."""
    return 1.0 + HELIUM_N_MINUS_1_AT_1_BAR * pressure_bar


def fires_threshold_counter(total_energy_gev, mass_gev, pressure_bar):
    """Return True when beta*n > 1 for helium at the supplied pressure."""
    beta = beta_relativistic(total_energy_gev, mass_gev)
    n_helium = helium_refractive_index(pressure_bar)
    return beta > (1.0 / n_helium)


def threshold_pressure_bar(total_energy_gev, mass_gev):
    """Minimum helium pressure in bar needed for Cherenkov emission."""
    beta = beta_relativistic(total_energy_gev, mass_gev)
    if beta <= 0.0:
        return np.nan
    return ((1.0 / beta) - 1.0) / HELIUM_N_MINUS_1_AT_1_BAR


def robust_firing_state(total_energy_gev, mass_gev, nominal_pressure_bar):
    """
    Robust firing state using the +/-10% pressure band:
      1.0: ON throughout the full band,
      0.0: OFF throughout the full band,
      0.5: threshold crossed inside the band.
    """
    pressure_low = nominal_pressure_bar * (1.0 - PRESSURE_RELATIVE_UNCERTAINTY)
    pressure_high = nominal_pressure_bar * (1.0 + PRESSURE_RELATIVE_UNCERTAINTY)

    if fires_threshold_counter(total_energy_gev, mass_gev, pressure_low):
        return 1.0
    if not fires_threshold_counter(total_energy_gev, mass_gev, pressure_high):
        return 0.0
    return 0.5


def state_label(state):
    """Readable name for a robust firing state."""
    if state == 1.0:
        return "ON"
    if state == 0.0:
        return "OFF"
    return "TURN-ON"


def pretty_pressure_mbar(pressure_mbar):
    """Compact pressure label for heatmap tick labels."""
    if pressure_mbar >= 1000.0:
        return f"{pressure_mbar / 1000.0:.1f}k"
    if pressure_mbar >= 100.0:
        return f"{pressure_mbar:.0f}"
    return f"{pressure_mbar:.1f}"


# ============================================================
# C1 input: supplied helium beam-plan numbers
# ============================================================
# C1 uses these exact values. The supplied set pressure tracks below the
# supplied pion threshold at each energy point.
# ============================================================

beam_plan = pd.DataFrame(
    {
        "Energy (GeV)": [6, 10, 20, 30, 40, 60, 80, 100, 120],
        "Pion threshold from table (mbar)": [
            7732, 2780, 696, 309, 174, 77, 43, 28, 19,
        ],
        "C1 supplied set pressure (mbar)": [
            3866, 2317, 350, 150, 80, 40, 20, 15, 10,
        ],
    }
)

beam_plan["Pion threshold from table (bar)"] = (
    beam_plan["Pion threshold from table (mbar)"] / 1000.0
)
beam_plan["C1 supplied set pressure (bar)"] = (
    beam_plan["C1 supplied set pressure (mbar)"] / 1000.0
)

# C1 follows the pion curve at a different fractional amount for each beam
# energy. Reuse that SAME fraction to make C2 follow the proton threshold.
beam_plan["C1 / pion-threshold fraction"] = (
    beam_plan["C1 supplied set pressure (bar)"]
    / beam_plan["Pion threshold from table (bar)"]
)

beam_plan["Calculated kaon threshold (bar)"] = [
    threshold_pressure_bar(energy, MASSES_GEV["K+"])
    for energy in beam_plan["Energy (GeV)"]
]
beam_plan["Calculated proton threshold (bar)"] = [
    threshold_pressure_bar(energy, MASSES_GEV["proton"])
    for energy in beam_plan["Energy (GeV)"]
]

# C2 follows the PROTON line at the same relative offset that C1 follows
# the PION line:
#
# P_C2 / P_p = P_C1 / P_pi(table)
beam_plan["C2 proton-following set pressure (bar)"] = (
    beam_plan["C1 / pion-threshold fraction"]
    * beam_plan["Calculated proton threshold (bar)"]
)
beam_plan["C2 proton-following set pressure (mbar)"] = (
    1000.0 * beam_plan["C2 proton-following set pressure (bar)"]
)

# Keep mbar copies of threshold curves for direct plotting / spreadsheet use.
for column in [
    "Calculated kaon threshold (bar)",
    "Calculated proton threshold (bar)",
]:
    beam_plan[column.replace("(bar)", "(mbar)")] = 1000.0 * beam_plan[column]

# +/-10% bands for both counter operating pressures.
for counter in [
    "C1 supplied set pressure",
    "C2 proton-following set pressure",
]:
    beam_plan[f"{counter} low (bar)"] = (
        beam_plan[f"{counter} (bar)"]
        * (1.0 - PRESSURE_RELATIVE_UNCERTAINTY)
    )
    beam_plan[f"{counter} high (bar)"] = (
        beam_plan[f"{counter} (bar)"]
        * (1.0 + PRESSURE_RELATIVE_UNCERTAINTY)
    )
    beam_plan[f"{counter} low (mbar)"] = (
        1000.0 * beam_plan[f"{counter} low (bar)"]
    )
    beam_plan[f"{counter} high (mbar)"] = (
        1000.0 * beam_plan[f"{counter} high (bar)"]
    )


# ============================================================
# Compute robust one-counter and two-counter firing signatures
# ============================================================

particle_order = ["e+", "pi+", "K+", "proton"]
summary_rows = []

for _, plan_row in beam_plan.iterrows():
    energy = float(plan_row["Energy (GeV)"])
    c1_bar = float(plan_row["C1 supplied set pressure (bar)"])
    c2_bar = float(plan_row["C2 proton-following set pressure (bar)"])

    for particle in particle_order:
        mass = MASSES_GEV[particle]

        c1_state = robust_firing_state(energy, mass, c1_bar)
        c2_state = robust_firing_state(energy, mass, c2_bar)

        # Use a binary signature only when both counter states are stable over
        # the entire pressure uncertainty band.
        if c1_state == 0.5 or c2_state == 0.5:
            signature = "??"
            signature_code = np.nan
        else:
            c1_bit = int(c1_state)
            c2_bit = int(c2_state)
            signature = f"{c1_bit}{c2_bit}"
            signature_code = 2 * c1_bit + c2_bit

        summary_rows.append(
            {
                "Energy (GeV)": energy,
                "Particle": particle,
                "C1 supplied pressure (mbar)": plan_row[
                    "C1 supplied set pressure (mbar)"
                ],
                "C2 proton-following pressure (mbar)": plan_row[
                    "C2 proton-following set pressure (mbar)"
                ],
                "Calculated particle threshold (mbar)": (
                    1000.0 * threshold_pressure_bar(energy, mass)
                ),
                "C1 state (+/-10%)": state_label(c1_state),
                "C2 state (+/-10%)": state_label(c2_state),
                "C1C2 signature": signature,
                "C1C2 signature code": signature_code,
            }
        )

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# Pivot tables for the heatmaps
# ============================================================

energies = beam_plan["Energy (GeV)"].tolist()

# One counter only: 0=OFF, 0.5=TURN-ON, 1=ON.
c1_heatmap = (
    summary_df.assign(
        C1_code=summary_df["C1 state (+/-10%)"].map(
            {"OFF": 0.0, "TURN-ON": 0.5, "ON": 1.0}
        )
    )
    .pivot(index="Particle", columns="Energy (GeV)", values="C1_code")
    .reindex(index=particle_order, columns=energies)
)

c1_annotation = (
    summary_df.pivot(
        index="Particle", columns="Energy (GeV)", values="C1 state (+/-10%)"
    )
    .reindex(index=particle_order, columns=energies)
)

# Two-counter code: 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3.
two_counter_heatmap = (
    summary_df.pivot(
        index="Particle", columns="Energy (GeV)", values="C1C2 signature code"
    )
    .reindex(index=particle_order, columns=energies)
)

two_counter_annotation = (
    summary_df.pivot(
        index="Particle", columns="Energy (GeV)", values="C1C2 signature"
    )
    .reindex(index=particle_order, columns=energies)
)

c1_pressure_mbar = beam_plan["C1 supplied set pressure (mbar)"].to_numpy()
c2_pressure_mbar = beam_plan[
    "C2 proton-following set pressure (mbar)"
].to_numpy()

c1_tick_labels = [
    f"{energy:g}\nC1={pretty_pressure_mbar(c1)}"
    for energy, c1 in zip(energies, c1_pressure_mbar)
]

two_counter_tick_labels = [
    f"{energy:g}\n{pretty_pressure_mbar(c1)} / {pretty_pressure_mbar(c2)}"
    for energy, c1, c2 in zip(energies, c1_pressure_mbar, c2_pressure_mbar)
]


# ============================================================
# Heatmap figure: one supplied counter versus two counters
# ============================================================

# 0=OFF, 0.5=threshold overlap / ambiguous, 1=ON.
state_cmap = ListedColormap(["#d9d9d9", "#ffd166", "#4daf4a"])
state_norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], state_cmap.N)

# 00, 01, 10, 11. The intended pattern contains 00, 01, and 11.
signature_cmap = ListedColormap(["#d9d9d9", "#4daf4a", "#74a9cf", "#756bb1"])
signature_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], signature_cmap.N)

fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

sns.heatmap(
    c1_heatmap,
    cmap=state_cmap,
    norm=state_norm,
    annot=c1_annotation,
    fmt="",
    cbar=False,
    linewidths=0.6,
    linecolor="black",
    ax=axes[0],
)
axes[0].set_title(
    "One counter: supplied C1 values following the pion curve",
    fontsize=13,
)
axes[0].set_xlabel("Beam energy [GeV] / C1 pressure [mbar]")
axes[0].set_ylabel("Particle hypothesis")
axes[0].set_xticklabels(c1_tick_labels, rotation=0)
axes[0].set_yticklabels(["e+", "pi+", "K+", "proton"], rotation=0)

sns.heatmap(
    two_counter_heatmap,
    cmap=signature_cmap,
    norm=signature_norm,
    annot=two_counter_annotation,
    fmt="",
    cbar=False,
    linewidths=0.6,
    linecolor="black",
    ax=axes[1],
)
axes[1].set_title(
    "Two counters: C1 pion-following + C2 proton-following",
    fontsize=13,
)
axes[1].set_xlabel("Beam energy [GeV] / C1,C2 pressures [mbar]")
axes[1].set_ylabel("")
axes[1].set_xticklabels(two_counter_tick_labels, rotation=0)
axes[1].set_yticklabels(["e+", "pi+", "K+", "proton"], rotation=0)

one_counter_legend = [
    Patch(facecolor="#4daf4a", edgecolor="black", label="ON across +/-10%"),
    Patch(facecolor="#d9d9d9", edgecolor="black", label="OFF across +/-10%"),
    Patch(facecolor="#ffd166", edgecolor="black", label="Threshold in +/-10% band"),
]
two_counter_legend = [
    Patch(facecolor="#756bb1", edgecolor="black", label="11: electron-like"),
    Patch(facecolor="#4daf4a", edgecolor="black", label="01: pion / kaon-like"),
    Patch(facecolor="#d9d9d9", edgecolor="black", label="00: proton-like"),
    Patch(facecolor="#74a9cf", edgecolor="black", label="10: C1 only"),
]

axes[0].legend(
    handles=one_counter_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    fontsize=9,
    frameon=True,
)
axes[1].legend(
    handles=two_counter_legend,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    fontsize=9,
    frameon=True,
)

fig.suptitle(
    "Helium Threshold-Cherenkov PID: Pion versus Proton Separation",
    fontsize=16,
)
fig.tight_layout(rect=[0, 0.08, 1, 0.94])

heatmap_path = os.path.join(
    SAVE_DIR,
    "helium_one_and_two_counter_pi_proton_heatmaps.pdf",
)
fig.savefig(heatmap_path, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Pressure schedule plot
# ============================================================

fig, axis = plt.subplots(figsize=(12, 8))

energy_array = beam_plan["Energy (GeV)"].to_numpy()
pion_threshold_mbar = beam_plan["Pion threshold from table (mbar)"].to_numpy()
kaon_threshold_mbar = beam_plan["Calculated kaon threshold (mbar)"].to_numpy()
proton_threshold_mbar = beam_plan["Calculated proton threshold (mbar)"].to_numpy()
c1_mbar = beam_plan["C1 supplied set pressure (mbar)"].to_numpy()
c2_mbar = beam_plan["C2 proton-following set pressure (mbar)"].to_numpy()

axis.plot(
    energy_array,
    pion_threshold_mbar,
    marker="o",
    linewidth=2.0,
    label="Pion threshold (supplied table)",
)
axis.plot(
    energy_array,
    kaon_threshold_mbar,
    marker="o",
    linewidth=1.7,
    label="Kaon threshold (calculated)",
)
axis.plot(
    energy_array,
    proton_threshold_mbar,
    marker="o",
    linewidth=1.7,
    label="Proton threshold (calculated)",
)

axis.plot(
    energy_array,
    c1_mbar,
    linestyle="--",
    marker="s",
    linewidth=2.4,
    label="C1: supplied pion-following set pressure",
)
axis.plot(
    energy_array,
    c2_mbar,
    linestyle="--",
    marker="^",
    linewidth=2.4,
    label="C2: proton-following set pressure",
)

axis.fill_between(
    energy_array,
    c1_mbar * (1.0 - PRESSURE_RELATIVE_UNCERTAINTY),
    c1_mbar * (1.0 + PRESSURE_RELATIVE_UNCERTAINTY),
    alpha=0.20,
    label="C1 +/-10%",
)
axis.fill_between(
    energy_array,
    c2_mbar * (1.0 - PRESSURE_RELATIVE_UNCERTAINTY),
    c2_mbar * (1.0 + PRESSURE_RELATIVE_UNCERTAINTY),
    alpha=0.20,
    label="C2 +/-10%",
)

axis.set_yscale("log")
axis.set_xlabel("Beam energy [GeV]")
axis.set_ylabel("Helium pressure [mbar]")
axis.set_title(
    "Counter Settings: C1 Follows Pions and C2 Follows Protons"
)
axis.minorticks_on()
axis.tick_params(axis="both", which="both", direction="in")
axis.grid(True, which="major", linestyle="--", alpha=0.35)
axis.grid(True, which="minor", linestyle=":", alpha=0.20)
axis.legend(fontsize=8, ncol=2)
fig.tight_layout()

pressure_plot_path = os.path.join(
    SAVE_DIR,
    "helium_pi_proton_counter_pressure_schedule.pdf",
)
fig.savefig(pressure_plot_path, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Save workbook and readable summary
# ============================================================

excel_path = os.path.join(
    SAVE_DIR,
    "helium_pi_proton_counter_pid_tables.xlsx",
)

with pd.ExcelWriter(excel_path) as writer:
    beam_plan.to_excel(writer, sheet_name="Pressure plan", index=False)
    summary_df.to_excel(writer, sheet_name="All firing states", index=False)
    c1_heatmap.to_excel(writer, sheet_name="C1 heatmap numeric")
    c1_annotation.to_excel(writer, sheet_name="C1 heatmap labels")
    two_counter_heatmap.to_excel(writer, sheet_name="C1C2 heatmap numeric")
    two_counter_annotation.to_excel(writer, sheet_name="C1C2 signatures")

summary_path = os.path.join(
    SAVE_DIR,
    "helium_pi_proton_counter_pid_summary.txt",
)

with open(summary_path, "w") as output_file:
    output_file.write("Helium one- and two-counter pion/proton PID summary\n")
    output_file.write("=" * 76 + "\n\n")
    output_file.write(
        "C1 uses the supplied helium set pressures, which follow below the pion threshold.\n"
    )
    output_file.write(
        "C2 follows below the proton threshold using the same fractional offset:\n"
    )
    output_file.write("    C2/Pp = C1/Ppi(table)\n")
    output_file.write(
        f"Every ON/OFF decision is required to stay stable across +/-{100 * PRESSURE_RELATIVE_UNCERTAINTY:.0f}% pressure.\n\n"
    )
    output_file.write("Pressure plan:\n")
    output_file.write(beam_plan.to_string(index=False))
    output_file.write("\n\nFiring states:\n")
    output_file.write(summary_df.to_string(index=False))
    output_file.write("\n")


# ============================================================
# Console summary
# ============================================================

print("\nHelium one- and two-counter pion/proton PID study finished.\n")
print(f"Heatmap PDF:       {heatmap_path}")
print(f"Pressure plot:     {pressure_plot_path}")
print(f"Excel workbook:    {excel_path}")
print(f"Text summary:      {summary_path}")
print("\nCounter definition:")
print("  C1 = supplied helium set pressure (pion-following curve)")
print("  C2 = (C1 / Ppi_table) * Pproton_threshold")
print("\nExpected robust signatures:")
print("  e+      -> 11")
print("  pi+     -> 01")
print("  K+      -> 01")
print("  proton  -> 00")
