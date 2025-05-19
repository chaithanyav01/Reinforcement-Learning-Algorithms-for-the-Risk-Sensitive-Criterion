import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import os

# === Matplotlib Style: Academic / Publication ===
mpl.rcParams.update({
    'font.size': 14,
    'font.family': 'Times New Roman',
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

# === File Paths ===
folder = "ppo_results_beta=5"
train_path = os.path.join(folder, "training_rewards.csv")
inference_files = [
    "inference_results_heavy_rain.csv",
    "inference_results_no_rain.csv",
    "inference_results_slight_rain.csv",
    "inference_results_town02.csv",
    "inference_results_town03.csv"
]

# === Load and Smooth Training Data ===
train_df = pd.read_csv(train_path).head(7000)
train_df['normalized_reward'] = (train_df['reward']).rolling(window=800, min_periods=1).mean()

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(10, 6))

# Shadow effect for better visibility
shadow_effect = [pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()]

# --- Plot Training Curve ---
ax.plot(train_df.index, train_df['normalized_reward'],
        label="Training", color='#1f77b4', linestyle='-', path_effects=shadow_effect)

# --- Inference Curves ---
start_index = train_df.index.max() + 1
max_inf_length = 0
inference_dfs = []

for file in inference_files:
    path = os.path.join(folder, file)
    inf_df = pd.read_csv(path)
    inf_df['normalized_cumreward'] = (inf_df['cumulative_reward']).rolling(window=100, min_periods=1).mean()
    inference_dfs.append((file, inf_df))
    max_inf_length = max(max_inf_length, len(inf_df))

x_vals = list(range(start_index, start_index + max_inf_length))
color_map = plt.get_cmap("tab10")

for i, (file, inf_df) in enumerate(inference_dfs):
    y_vals = inf_df['normalized_cumreward'].tolist()
    if len(y_vals) < max_inf_length:
        y_vals += [float('nan')] * (max_inf_length - len(y_vals))
    label = file.replace("inference_results_", "").replace(".csv", "").replace("_", " ").capitalize()
    ax.plot(x_vals, y_vals, label=label, linestyle='--', color=color_map(i), path_effects=shadow_effect)

# === Axis Labels ===
ax.set_xlabel("Episodes", labelpad=10)
ax.set_ylabel("Rewards (Moving Average)", labelpad=10)

# === Title ===
ax.set_title("Training and Inference Rewards for PPO (β=5)", fontweight='bold', fontsize=16, pad=12)

# === Legend ===
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=12)

# === Grid & Spines ===
ax.grid(True)                    # Enable grid on all axes
ax.spines['top'].set_visible(True)    # Show top spine
ax.spines['right'].set_visible(True)  # Show right spine

# === Final Touches ===
fig.tight_layout()

# === Save Plot ===
plt.savefig("train_inf_plots/ppo_beta=5.png", dpi=300, bbox_inches='tight')
plt.close()
