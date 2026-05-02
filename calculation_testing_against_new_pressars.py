import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================================================
# Physics constants and setup
# ======================================================

masses = {
    # leptons
    "e⁺": 0.000511,
    "μ⁺": 0.105658,

    # pseudoscalar mesons
    "π⁺": 0.13957039,
    "π⁻": 0.13957039,
    "K⁺": 0.493677,
    "K⁻": 0.493677,
    "η": 0.547862,
    "η′": 0.95778,

    # vector mesons
    "ρ": 0.77526,
    "ω": 0.78265,
    "ϕ": 1.019461,
    "K* (892)": 0.89555,

    # baryons
    "proton": 0.938272,
    "n": 0.939565,
    "Λ": 1.115683,
    "Σ⁺": 1.18937,
    "Σ⁻": 1.19745,
    "Ξ⁻": 1.32171,
}

# Use DISCRETE energies for tables & heatmaps
energies_discrete = [5, 10, 20, 30, 40, 60, 80, 100, 120, 160]

# Use CONTINUOUS energies for threshold pressure curve
energies_continuous = np.linspace(1, 200, 800)

# New recommended pressures
new_pressures = [0.2, 0.5, 0.7]
old_pressures = [0.3, 0.6, 1.0]

# Helium refractive index offset at 1 bar
# n - 1 = 35.53e-6 at 1 bar
n1_minus_1 = 3.553e-5

save_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Calculations2"
os.makedirs(save_dir, exist_ok=True)


# ======================================================
# Physics helper functions
# ======================================================

def gamma_relativistic(E, m):
    if E <= m:
        return 1.0
    return E / m


def beta_relativistic(E, m):
    if E <= m:
        return 0.0
    return np.sqrt(1.0 - (m / E) ** 2)


def momentum(E, m):
    if E <= m:
        return 0.0
    return np.sqrt(E ** 2 - m ** 2)


def refractive_index_at_pressure(P_bar):
    """
    Helium refractive index at pressure P_bar.
    Assumes n - 1 scales linearly with pressure.
    """
    return 1.0 + n1_minus_1 * P_bar


def cherenkov_condition(E, m, P_bar):
    beta = beta_relativistic(E, m)
    n = refractive_index_at_pressure(P_bar)
    return beta > 1.0 / n


def pressure_required_for_cherenkov(E, m):
    """
    Returns threshold pressure in bar for a particle with total energy E.
    """
    beta = beta_relativistic(E, m)

    if beta <= 0:
        return np.nan

    n_min = 1.0 / beta
    return (n_min - 1.0) / n1_minus_1


def threshold_momentum_at_pressure(m, P_bar):
    """
    Returns Cherenkov threshold momentum in GeV/c for mass m at pressure P_bar.

    Threshold condition:
        beta_thr = 1/n

    Then:
        p_thr = m * beta / sqrt(1 - beta^2)

    Equivalent:
        p_thr = m / sqrt(n^2 - 1)
    """
    n = refractive_index_at_pressure(P_bar)
    return m / np.sqrt(n ** 2 - 1.0)


# ======================================================
# Build discrete-energy table + text file output
# ======================================================
# ======================================================
# Pressure settings
# ======================================================

# Old pressures used in your previous calculation
old_pressures = [0.3, 0.6, 1.0]  # bar

# New proposed pressures
new_pressures = [0.2, 0.5, 0.7]  # bar

# Main pressures used for the firing table and heatmap
pressures = old_pressures

rows = []
text_lines = []
text_lines.append("Cherenkov Firing Summary\n")
text_lines.append("====================================\n")
text_lines.append(f"Pressures used: {pressures} bar\n")

for particle, m in masses.items():
    text_lines.append(f"\n===== {particle} (mass = {m:.6f} GeV) =====\n")

    for E in energies_discrete:
        b = beta_relativistic(E, m)
        g = gamma_relativistic(E, m)
        p = momentum(E, m)

        fire_status = {P: cherenkov_condition(E, m, P) for P in pressures}

        rows.append({
            "Particle": particle,
            "Energy (GeV)": E,
            "beta": round(b, 6),
            "gamma": round(g, 6),
            "momentum (GeV/c)": round(p, 6),
            **{
                f"Fire @ {P} bar": ("Yes" if fire_status[P] else "No")
                for P in pressures
            }
        })

        status_string = ", ".join([
            f"{P} bar: {'FIRE' if fire_status[P] else 'NO'}"
            for P in pressures
        ])

        text_lines.append(
            f"E={E:6.2f} GeV | "
            f"β={b:.5f} | "
            f"γ={g:.3f} | "
            f"p={p:.3f} GeV/c | "
            f"{status_string}"
        )

# Save text file
text_path = os.path.join(save_dir, "cherenkov_verbose_output_020_050_070.txt")
with open(text_path, "w") as f:
    f.write("\n".join(text_lines))

print(f"✅ Text summary saved to:\n{text_path}")


# ======================================================
# Save Excel table
# ======================================================

df = pd.DataFrame(rows)

excel_path = os.path.join(save_dir, "cherenkov_full_particle_table_020_050_070.xlsx")
df.to_excel(excel_path, index=False)

print(f"✅ Excel table saved to:\n{excel_path}")


# ======================================================
# Save threshold momentum table
# ======================================================

thr_rows = []

for particle, m in masses.items():
    row = {"Particle": particle, "mass [GeV]": m}

    for P in pressures:
        p_thr = threshold_momentum_at_pressure(m, P)
        row[f"p_thr @ {P} bar [GeV/c]"] = p_thr

    thr_rows.append(row)

df_thr = pd.DataFrame(thr_rows)

thr_table_path = os.path.join(save_dir, "cherenkov_threshold_momenta_020_050_070.xlsx")
df_thr.to_excel(thr_table_path, index=False)

print(f"✅ Threshold momentum table saved to:\n{thr_table_path}")


# ======================================================
# Print useful threshold momenta
# ======================================================

print("\n======================================================")
print("Threshold momenta at selected pressures")
print("======================================================")

for particle in ["μ⁺", "π⁺", "K⁺", "proton"]:
    m = masses[particle]
    print(f"\n{particle}:")
    for P in pressures:
        p_thr = threshold_momentum_at_pressure(m, P)
        print(f"  {P:.1f} bar : {p_thr:8.2f} GeV/c")


# ======================================================
# Heatmap Visualization: DISCRETE energies
# ======================================================

df_plot = df.copy()

for col in df_plot.columns:
    if "Fire @" in col:
        df_plot[col] = df_plot[col].map({"Yes": 1, "No": 0})

fig, axes = plt.subplots(1, len(pressures), figsize=(18, 10), sharey=True)

for ax, P in zip(axes, pressures):
    col = f"Fire @ {P} bar"
    pivot = df_plot.pivot(index="Particle", columns="Energy (GeV)", values=col)

    sns.heatmap(
        pivot,
        cmap=["#ff6961", "#77dd77"],
        cbar=False,
        linewidths=0.4,
        linecolor="black",
        ax=ax
    )

    ax.set_title(f"{P} bar", fontsize=14)
    ax.set_xlabel("Beam Energy [GeV]")

axes[0].set_ylabel("Particle")

plt.suptitle("Cherenkov Firing Map — Helium Gas", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

heatmap_path = os.path.join(save_dir, "cherenkov_heatmap_020_050_070.pdf")
plt.savefig(heatmap_path)
plt.close()

print(f"✅ Heatmap saved to:\n{heatmap_path}")


# ======================================================
# Pressure vs Energy Curve
# ======================================================

plt.figure(figsize=(10, 7))

for particle, m in masses.items():
    Pmin = [pressure_required_for_cherenkov(E, m) for E in energies_continuous]
    plt.plot(energies_continuous, Pmin, label=particle, linewidth=1.0)

# Horizontal pressure threshold lines
for P in pressures:
    plt.axhline(P, color="black", linestyle="--", alpha=0.65)
    plt.text(
        202,
        P,
        f"{P:.1f} bar",
        va="center",
        ha="left",
        fontsize=9
    )

plt.minorticks_on()
plt.tick_params(axis="both", which="both", direction="in")

plt.grid(True, which="major", linestyle="--", alpha=0.35)
plt.grid(True, which="minor", linestyle=":", alpha=0.25)

plt.ylim(0, 5)
plt.xlim(0, 210)

plt.xlabel("Beam Energy [GeV]")
plt.ylabel("Minimum Pressure Required [bar]")
plt.title("Cherenkov Threshold Pressure vs Energy — Helium")
plt.legend(ncol=2, fontsize=7)

pressure_plot_path = os.path.join(save_dir, "pressure_vs_energy_020_050_070.pdf")
plt.savefig(pressure_plot_path, bbox_inches="tight")
plt.close()

print(f"✅ Pressure vs energy plot saved to:\n{pressure_plot_path}")

# ======================================================
# Momentum threshold vs pressure plot
# Compare OLD pressures vs NEW proposed pressures
# ======================================================

new_pressures = [0.2, 0.5, 0.7]
old_pressures = [0.3, 0.6, 1.0]

particles_for_momentum_plot = {
    "μ⁺": masses["μ⁺"],
    "π⁺": masses["π⁺"],
    "K⁺": masses["K⁺"],
    "proton": masses["proton"],
}

pressure_scan = np.linspace(0.02, 3.0, 800)

plt.figure(figsize=(12, 8))

# Draw threshold momentum curves
for particle, m in particles_for_momentum_plot.items():
    p_thr_curve = [
        threshold_momentum_at_pressure(m, P)
        for P in pressure_scan
    ]

    plt.plot(
        pressure_scan,
        p_thr_curve,
        linewidth=2.0,
        label=particle
    )

# ------------------------------------------------------
# OLD pressure vertical lines
# ------------------------------------------------------
for P in old_pressures:
    plt.axvline(
        P,
        color="gray",
        linestyle=":",
        linewidth=1.8,
        alpha=0.85
    )

    plt.text(
        P,
        292,
        f"old {P:.1f}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
        color="gray"
    )

# ------------------------------------------------------
# NEW pressure vertical lines
# ------------------------------------------------------
for P in new_pressures:
    plt.axvline(
        P,
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85
    )

    plt.text(
        P,
        292,
        f"new {P:.1f}",
        rotation=90,
        va="top",
        ha="left",
        fontsize=9,
        color="black"
    )

# ------------------------------------------------------
# Mark threshold points for NEW pressures
# ------------------------------------------------------
for particle, m in particles_for_momentum_plot.items():
    for P in new_pressures:
        p_thr = threshold_momentum_at_pressure(m, P)

        if p_thr <= 300:
            plt.scatter(P, p_thr, s=45)

            plt.text(
                P + 0.025,
                p_thr,
                f"{p_thr:.1f}",
                fontsize=8,
                va="center"
            )

# ------------------------------------------------------
# Mark threshold points for OLD pressures
# ------------------------------------------------------
for particle, m in particles_for_momentum_plot.items():
    for P in old_pressures:
        p_thr = threshold_momentum_at_pressure(m, P)

        if p_thr <= 300:
            plt.scatter(P, p_thr, s=35, marker="x")

            plt.text(
                P + 0.025,
                p_thr,
                f"{p_thr:.1f}",
                fontsize=8,
                va="center",
                color="gray"
            )

# Dummy legend entries for old/new pressure lines
plt.plot([], [], color="black", linestyle="--", linewidth=1.8, label="new pressures")
plt.plot([], [], color="gray", linestyle=":", linewidth=1.8, label="old pressures")

plt.minorticks_on()
plt.tick_params(axis="both", which="both", direction="in")

plt.grid(True, which="major", linestyle="--", alpha=0.35)
plt.grid(True, which="minor", linestyle=":", alpha=0.25)

plt.xlim(0, 3.0)
plt.ylim(0, 300)

plt.xlabel("Pressure [bar]")
plt.ylabel("Threshold Momentum [GeV/c]")
plt.title("Cherenkov Threshold Momentum vs Pressure — Helium")
plt.legend(fontsize=10, ncol=2)

momentum_plot_path = os.path.join(
    save_dir,
    "threshold_momentum_vs_pressure_old_vs_new.pdf"
)

plt.savefig(momentum_plot_path, bbox_inches="tight")
plt.close()

print(f"✅ Old vs new momentum threshold plot saved to:\n{momentum_plot_path}")



# ======================================================
# Beam momentum firing pattern: old vs new pressures
# ======================================================

beam_momenta = [10,20, 30, 40, 60, 80, 100, 120, 160]

pid_particles = {
    "μ⁺": masses["μ⁺"],
    "π⁺": masses["π⁺"],
    "K⁺": masses["K⁺"],
    "proton": masses["proton"],
}

pattern_rows = []

def fires_at_momentum(p_beam, m, P_bar):
    n = refractive_index_at_pressure(P_bar)
    beta = p_beam / np.sqrt(p_beam**2 + m**2)
    return beta > 1.0 / n

for p_beam in beam_momenta:
    for particle, m in pid_particles.items():

        old_pattern = "".join([
            "1" if fires_at_momentum(p_beam, m, P) else "0"
            for P in old_pressures
        ])

        new_pattern = "".join([
            "1" if fires_at_momentum(p_beam, m, P) else "0"
            for P in new_pressures
        ])

        pattern_rows.append({
            "Momentum [GeV/c]": p_beam,
            "Particle": particle,
            "OLD pressures": str(old_pressures),
            "OLD pattern": old_pattern,
            "NEW pressures": str(new_pressures),
            "NEW pattern": new_pattern,
        })

df_patterns = pd.DataFrame(pattern_rows)

pattern_path = os.path.join(
    save_dir,
    "old_vs_new_firing_patterns.xlsx"
)

df_patterns.to_excel(pattern_path, index=False)

print(f"✅ Old vs new firing pattern table saved to:\n{pattern_path}")

print("\n======================================================")
print("Firing patterns")
print("Pattern order:")
print(f"  OLD = {old_pressures}")
print(f"  NEW = {new_pressures}")
print("1 = fires, 0 = does not fire")
print("======================================================")

for p_beam in beam_momenta:
    print(f"\nMomentum = {p_beam} GeV/c")
    sub = df_patterns[df_patterns["Momentum [GeV/c]"] == p_beam]

    for _, row in sub.iterrows():
        print(
            f"  {row['Particle']:7s} | "
            f"OLD {row['OLD pattern']} | "
            f"NEW {row['NEW pattern']}"
        )

# ======================================================
# Optional: zoomed momentum threshold plot
# ======================================================

plt.figure(figsize=(11, 8))

for particle, m in particles_for_momentum_plot.items():
    p_thr_curve = [
        threshold_momentum_at_pressure(m, P)
        for P in pressure_scan
    ]

    plt.plot(
        pressure_scan,
        p_thr_curve,
        linewidth=2.0,
        label=particle
    )

for P in pressures:
    plt.axvline(
        P,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65
    )

    plt.text(
        P,
        147,
        f"{P:.1f} bar",
        rotation=90,
        va="top",
        ha="right",
        fontsize=10,
        alpha=0.9
    )

for particle, m in particles_for_momentum_plot.items():
    for P in pressures:
        p_thr = threshold_momentum_at_pressure(m, P)

        if p_thr <= 150:
            plt.scatter(P, p_thr, s=35)

            plt.text(
                P + 0.025,
                p_thr,
                f"{particle}: {p_thr:.1f}",
                fontsize=8,
                va="center"
            )

plt.minorticks_on()
plt.tick_params(axis="both", which="both", direction="in")

plt.grid(True, which="major", linestyle="--", alpha=0.35)
plt.grid(True, which="minor", linestyle=":", alpha=0.25)

plt.xlim(0, 1.2)
plt.ylim(0, 150)

plt.xlabel("Pressure [bar]")
plt.ylabel("Threshold Momentum [GeV/c]")
plt.title("Cherenkov Threshold Momentum vs Pressure — Helium, Zoomed")
plt.legend(fontsize=11)

momentum_zoom_plot_path = os.path.join(save_dir, "threshold_momentum_vs_pressure_020_050_070_zoom_with_vertical_lines.pdf")
plt.savefig(momentum_zoom_plot_path, bbox_inches="tight")
plt.close()

print(f"✅ Zoomed momentum threshold plot saved to:\n{momentum_zoom_plot_path}")
