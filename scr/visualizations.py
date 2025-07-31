import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Load the datasets
word_stats_emot = pd.read_csv('data/merged/word_stats_emot.csv')
freq_stats_emot = pd.read_csv('data/merged/freq_stats_emot.csv')


# Regression Helper 
def add_regression(ax, x, y, color='red', label_prefix='Simple', threshold=2):
    if len(x) >= threshold and np.unique(x).size > 1:
        slope, intercept, r_value, _, _ = linregress(x, y)
        x_vals = np.sort(x)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color=color, linestyle='-', linewidth=1.5)
        ax.legend([f'{label_prefix} y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.2f}'],
                  loc='lower right', fontsize='x-small')


# Plot 1 
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# The min and max values from the paper DOI: 10.3758/s13428-012-0314-x
vmin = 1
vmax = 9

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

sc1 = axes[0].scatter(
    word_stats_emot['log_freq'], word_stats_emot['log_need'],
    s=20 + 380 * word_stats_emot['arousal_n'],
    c=word_stats_emot['valence'],
    cmap='viridis', alpha=0.25 + 0.75 * word_stats_emot['arousal_n'],
    edgecolors='none', vmin=vmin, vmax=vmax
)
axes[0].set_title('Group by Word: Arousal→Size/Opacity, Valence→Colour')
axes[0].set_ylabel('Log Need Odds')
fig.colorbar(sc1, ax=axes[0], label='Valence')
add_regression(axes[0], word_stats_emot['log_freq'], word_stats_emot['log_need'])

sc2 = axes[1].scatter(
    freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
    s=20 + 380 * freq_stats_emot['arousal_n'],
    c=freq_stats_emot['mean_valence'],
    cmap='viridis', alpha=0.25 + 0.75 * freq_stats_emot['arousal_n'],
    edgecolors='none', vmin=vmin, vmax=vmax
)
axes[1].set_title('Group by Freq: Arousal→Size/Opacity, Valence→Colour')
axes[1].set_xlabel('Log Frequency')
axes[1].set_ylabel('Log Need Odds')
fig.colorbar(sc2, ax=axes[1], label='Mean Valence')
add_regression(axes[1], freq_stats_emot['log_freq'], freq_stats_emot['log_need'])

for ax in axes:
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("output/plots/analysis_with_emotional_norm.png", dpi=300)

# Plot 2 
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

sc0 = axes[0, 0].scatter(word_stats_emot['log_freq'], word_stats_emot['log_need'],
                         c=word_stats_emot['valence'], cmap='coolwarm', alpha=0.7,
                         vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Word: Valence")
fig.colorbar(sc0, ax=axes[0, 0], label="Valence")

sc1 = axes[1, 0].scatter(word_stats_emot['log_freq'], word_stats_emot['log_need'],
                         c=word_stats_emot['arousal'], cmap='coolwarm', alpha=0.7,
                         vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Word: Arousal")
fig.colorbar(sc1, ax=axes[1, 0], label="Arousal")

sc2 = axes[0, 1].scatter(freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
                         c=freq_stats_emot['mean_valence'], cmap='coolwarm', alpha=0.8,
                         vmin=vmin, vmax=vmax)
axes[0, 1].set_title("Freq: Valence")
fig.colorbar(sc2, ax=axes[0, 1], label="Mean Valence")

sc3 = axes[1, 1].scatter(freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
                         c=freq_stats_emot['mean_arousal'], cmap='coolwarm', alpha=0.8,
                         vmin=vmin, vmax=vmax)
axes[1, 1].set_title("Freq: Arousal")
fig.colorbar(sc3, ax=axes[1, 1], label="Mean Arousal")

for ax in axes[1, :]:
    ax.set_xlabel("Log Frequency")
for ax in axes[:, 0]:
    ax.set_ylabel("Log Need Odds")
for row in axes:
    for ax in row:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("output/plots/word_freq_split_valence_arousal.png", dpi=300)

# Plot 3 
valence_labels = ['Low Valence', 'Mid Valence', 'High Valence']
arousal_labels = ['Low Arousal', 'Mid Arousal', 'High Arousal']
word_stats_emot['valence_group'] = pd.qcut(word_stats_emot['valence'], q=3, labels=valence_labels)
word_stats_emot['arousal_group'] = pd.qcut(word_stats_emot['arousal'], q=3, labels=arousal_labels)

fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True, sharey=True)

for i, valence_label in enumerate(valence_labels):
    for j, arousal_label in enumerate(arousal_labels):
        ax = axes[i, j]
        subset = word_stats_emot[
            (word_stats_emot['valence_group'] == valence_label) &
            (word_stats_emot['arousal_group'] == arousal_label)
        ]
        sc = ax.scatter(
            subset['log_freq'], subset['log_need'],
            s=20 + 380 * subset['arousal_n'],
            c=subset['valence'], cmap='viridis', alpha=0.7,
            edgecolors='none', vmin=vmin, vmax=vmax
        )
        ax.set_title(f'{valence_label} - {arousal_label}')
        add_regression(ax, subset['log_freq'], subset['log_need'])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

for ax in axes[-1, :]:
    ax.set_xlabel('Log Frequency')
for ax in axes[:, 0]:
    ax.set_ylabel('Log Need Odds')

plt.tight_layout()
plt.savefig("output/plots/split_valence_arousal_9plots.png", dpi=300)


# Plot 4
valence_colors = dict(zip(valence_labels, ['blue', 'green', 'red']))
arousal_sizes = dict(zip(arousal_labels, [50, 200, 500]))

fig, ax = plt.subplots(figsize=(12, 8))

for valence_label in valence_labels:
    for arousal_label in arousal_labels:
        subset = word_stats_emot[
            (word_stats_emot['valence_group'] == valence_label) &
            (word_stats_emot['arousal_group'] == arousal_label)
        ]
        ax.scatter(
            subset['log_freq'], subset['log_need'],
            s=arousal_sizes[arousal_label],
            c=valence_colors[valence_label],
            alpha=0.7, edgecolors='k', linewidths=0.5,
            label=f'{valence_label} - {arousal_label}'
        )

valid_data = word_stats_emot[word_stats_emot['need_odds'] > 0]
add_regression(ax, valid_data['log_freq'], valid_data['log_need'], color='black', label_prefix='Combined')

ax.set_xlabel('Log Frequency')
ax.set_ylabel('Log Need Odds')
ax.set_title('Combined Plot: Valence (Color) and Arousal (Size)')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Groups')

plt.tight_layout()
plt.savefig("output/plots/combined_3split_summary_plot.png", dpi=300)
