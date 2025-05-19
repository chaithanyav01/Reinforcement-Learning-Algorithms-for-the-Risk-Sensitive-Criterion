import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import os

# === Matplotlib Style: Academic / Publication with Times New Roman & consistent font sizes ===
mpl.rcParams.update({
    'font.size': 14,                          # base font size
    'font.family': 'Times New Roman',        # Times New Roman globally
    'axes.labelsize': 16,                     # axis labels size
    'axes.labelweight': 'bold',               # bold axis labels
    'axes.linewidth': 1.2,
    'axes.titlesize': 16,                     # title size
    'axes.titleweight': 'bold',               # bold title
    'legend.fontsize': 12,                    # legend font size
    'legend.frameon': False,
    'legend.title_fontsize': 14,
    'xtick.labelsize': 12,                    # tick label size
    'ytick.labelsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'grid.linestyle': ':',
    'grid.linewidth': 0.7
})

# === File Paths ===
folder = "ppo_results_beta=0"
train_path = os.path.join(folder, "training_rewards.csv")
inference_files = [
    "inference_results_heavy_rain.csv",
    "inference_results_no_rain.csv",
    "inference_results_slight_rain.csv",
    "inference_results_town02.csv",
    "inference_results_town03.csv"
]

# === Load Training Data ===
train_df = pd.read_csv(train_path)
train_df['normalized_reward'] = (train_df['reward'] / 500).rolling(window=1000, min_periods=1).mean()

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(10, 6))

# Shadow effect for visibility
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
    inf_df['normalized_cumreward'] = (inf_df['cumulative_reward'] / 800).rolling(window=100, min_periods=1).mean()
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
ax.set_xlabel("Episodes", labelpad=10, fontweight='bold', family='Times New Roman', fontsize=16)
ax.set_ylabel("Rewards (Moving Average)", labelpad=10, fontweight='bold', family='Times New Roman', fontsize=16)

# === Title ===
ax.set_title("Training and Inference Rewards for PPO (β=0)",
             fontweight='bold', fontsize=16, family='Times New Roman', pad=12)

# Set tick label font family, weight and size manually
for label in ax.get_xticklabels():
    label.set_fontweight('normal')
    label.set_family('Times New Roman')
    label.set_fontsize(12)

for label in ax.get_yticklabels():
    label.set_fontweight('normal')
    label.set_family('Times New Roman')
    label.set_fontsize(12)

# === Legend ===
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=12)
for text in legend.get_texts():
    text.set_family('Times New Roman')
    text.set_fontsize(12)

# === Grid & Spines ===
ax.grid(True)
ax.spines['top'].set_visible(True)     # Show top spine
ax.spines['right'].set_visible(True)   # Show right spine

# === Final Touches ===
fig.tight_layout()

# === Save Plot ===
output_path = "train_inf_plots/ppo_beta=0.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
