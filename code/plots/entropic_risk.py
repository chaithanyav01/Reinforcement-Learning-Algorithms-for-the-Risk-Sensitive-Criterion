import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.patheffects as pe
import os

# === Matplotlib Style: Academic / Publication ===
mpl.rcParams.update({
    'font.size': 14,
    'font.family': 'Times New Roman',  # Classic research paper font
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'legend.frameon': False,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'grid.linestyle': ':',
    'grid.linewidth': 0.7
})

# === Entropic Risk Function ===
def compute_entropic_risk(rewards, beta):
    return -1 / beta * np.log(np.mean(np.exp(-beta * rewards)))

# === Settings ===
beta_values = [5, 10]
algorithms = ['ppo', 'sac']
file_conditions = [
    ('no_rain', 'inference_results_no_rain.csv'),
    ('slight_rain', 'inference_results_slight_rain.csv'),
    ('heavy_rain', 'inference_results_heavy_rain.csv'),
    ('town02', 'inference_results_town02.csv'),
    ('town03', 'inference_results_town03.csv')
]
colors = sns.color_palette("tab10", 6)

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Shadow effect for plot lines
shadow_effect = [pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()]

# === Logging Setup ===
log_data = []

# === Plotting ===
color_idx = 0
for algo in algorithms:
    for beta in beta_values:
        folder = f"{algo}_results_beta={beta}"
        risk_list = []
        labels = []

        for label, filename in file_conditions:
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                rewards = df['cumulative_reward'].values
                risk = compute_entropic_risk(rewards, beta)
            else:
                risk = np.nan

            risk_list.append(risk)
            labels.append(label)
            log_data.append({
                "algorithm": algo.upper(),
                "beta": beta,
                "condition": label,
                "entropic_risk": risk
            })

        plt.plot(labels, risk_list, marker='o', label=f"{algo.upper()} β={beta}",
                 color=colors[color_idx], path_effects=shadow_effect)
        color_idx += 1

# === Axis Labels & Title ===
plt.xlabel("Modified Environment for Inference", labelpad=10)
plt.ylabel("Entropic Risk", labelpad=10)
plt.title("Entropic Risk vs Modified Environment during Inference for PPO and SAC", pad=12, fontweight='bold')

# === Legend ===
leg = plt.legend(title="Algorithm + Beta", loc='upper right')
leg.get_title().set_fontweight('bold')

# === Grid & Layout ===
plt.grid(True)
plt.tight_layout()

# === Save Plot ===
# os.makedirs("metric_plots", exist_ok=True)
plt.savefig("metric_plots/entropic_risk_vs_condition.png", dpi=300, bbox_inches='tight')
# plt.show()

# === Save Logs ===
# os.makedirs("logs", exist_ok=True)
log_df = pd.DataFrame(log_data)
log_df.to_csv("logs/entropic_risk_log_conditions.csv", index=False)
