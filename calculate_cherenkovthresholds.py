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

pressures = [0.3, 0.6, 1.0]
n1_minus_1 = 3.553e-5

save_dir = "/lustre/research/hep/akshriva/Dream-testbeam2-analysis/Calculations"
os.makedirs(save_dir, exist_ok=True)


# ======================================================
# Physics helper functions
# ======================================================

def gamma_relativistic(E, m):
    if E <= m: return 1.0
    return E / m

def beta_relativistic(E, m):
    if E <= m: return 0.0
    return np.sqrt(1 - (m/E)**2)

def momentum(E, m):
    if E <= m: return 0.0
    return np.sqrt(E**2 - m**2)

def refractive_index_at_pressure(P):
    return 1 + n1_minus_1 * P

def cherenkov_condition(E, m, P):
    beta = beta_relativistic(E, m)
    n = refractive_index_at_pressure(P)
    return beta > 1/n

def pressure_required_for_cherenkov(E, m):
    beta = beta_relativistic(E, m)
    if beta <= 0: return np.nan
    n_min = 1/beta
    return (n_min - 1) / n1_minus_1


# ======================================================
# Build discrete-energy table + text file output
# ======================================================

rows = []
text_lines = []
text_lines.append("Cherenkov Firing Summary\n")
text_lines.append("====================================\n")

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
            "momentum (GeV)": round(p, 6),
            **{f"Fire @ {P} bar": ("Yes" if fire_status[P] else "No") for P in pressures}
        })

        status_string = ", ".join([
            f"{P}bar: {'FIRE' if fire_status[P] else 'NO'}" 
            for P in pressures
        ])

        text_lines.append(
            f"E={E:6.2f} GeV | β={b:.5f} | γ={g:.3f} | p={p:.3f} GeV | {status_string}"
        )

# Save text file
text_path = os.path.join(save_dir, "cherenkov_verbose_output.txt")
with open(text_path, "w") as f:
    f.write("\n".join(text_lines))

print(f"✅ Text summary saved to:\n{text_path}")


# ======================================================
# Save Excel table
# ======================================================

df = pd.DataFrame(rows)
excel_path = os.path.join(save_dir, "cherenkov_full_particle_table.xlsx")
df.to_excel(excel_path, index=False)
print(f"✅ Excel table saved to:\n{excel_path}")


# ======================================================
# Heatmap Visualization (DISCRETE energies)
# ======================================================

df_plot = df.copy()
for col in df_plot.columns:
    if "Fire @" in col:
        df_plot[col] = df_plot[col].map({"Yes": 1, "No": 0})

pressure_labels = [f"{P} bar" for P in pressures]
fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)

for ax, P in zip(axes, pressures):
    col = f"Fire @ {P} bar"
    pivot = df_plot.pivot(index="Particle", columns="Energy (GeV)", values=col)

    sns.heatmap(
        pivot, cmap=["#ff6961", "#77dd77"], cbar=False,
        linewidths=0.4, linecolor="black", ax=ax
    )

    ax.set_title(f"{P} bar", fontsize=14)
    ax.set_xlabel("Beam Energy [GeV]")

axes[0].set_ylabel("Particle")
plt.suptitle("Cherenkov Firing Map — Helium Gas (Discrete Energies)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

heatmap_path = os.path.join(save_dir, "cherenkov_heatmap_full_particles.pdf")
plt.savefig(heatmap_path)
plt.close()
print(f"✅ Heatmap saved to:\n{heatmap_path}")


# ======================================================
# Pressure vs Energy Curve (continuous energies)
# ======================================================

plt.figure(figsize=(10, 7))

for particle, m in masses.items():
    Pmin = [pressure_required_for_cherenkov(E, m) for E in energies_continuous]
    plt.plot(energies_continuous, Pmin, label=particle, linewidth=1.0)

# pressure threshold lines
for P in pressures:
    plt.axhline(P, color="black", linestyle="--", alpha=0.6)

# Minor ticks
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in')
plt.grid(True, which='major', linestyle='--', alpha=0.35)
plt.grid(True, which='minor', linestyle=':', alpha=0.25)

plt.ylim(0, 5)
plt.xlabel("Beam Energy (GeV)")
plt.ylabel("Minimum Pressure Required (bar)")
plt.title("Cherenkov Threshold Pressure vs Energy (Helium)")
plt.legend(ncol=2, fontsize=7)

pressure_plot_path = os.path.join(save_dir, "pressure_vs_energy_full_particles.pdf")
plt.savefig(pressure_plot_path)
plt.close()

print(f"✅ Pressure vs Energy plot saved to:\n{pressure_plot_path}")
