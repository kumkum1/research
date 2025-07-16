import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from metrics import word_stats, freq_stats, recall_df         
from combine_data import cleaned_df


# print(cleaned_df)
#freq_stats -> [1893 rows x 6 columns]

emot_df = pd.read_csv('emot_28724.csv')
emot_df.rename(columns=lambda x: x.lower().strip(), inplace=True)
emot_df = emot_df[['word', 'valence', 'arousal']]

merged_df = pd.merge(word_stats, emot_df, on='word', how='left')
merged_df.to_csv('combined_with_emotion.csv', index=False)

# data with only complete rows for emotion analysis
word_stats_emot = merged_df.dropna(subset=['valence', 'arousal'])
print(word_stats_emot)

# --------------------------------------------- ##

# recall_df['word'] = recall_df['word'].replace('mln', 'million')
# recall_df['word'] = recall_df['word'].replace('pct', 'percent')

recalled_with_emotion = (
    recall_df.merge(emot_df, on='word', how='left')
)

# print(recall_df)
# print(recalled_with_emotion)

freq_stats_emot = recalled_with_emotion.groupby('frequency').agg(
    total=('is_recalled', 'count'),
    recalls=('is_recalled', 'sum'),
    mean_valence=('valence', 'mean'),
    mean_arousal=('arousal', 'mean')
).reset_index()
freq_stats_emot['recall_probability'] = freq_stats_emot['recalls'] / freq_stats_emot['total']
freq_stats_emot['need_odds'] = freq_stats_emot['recall_probability'].transform(lambda x: x/(1-x))
freq_stats_emot.dropna(subset=['mean_valence', 'mean_arousal'], inplace=True)

print(freq_stats_emot)
# print(freq_stats)

# --- Normalization Helper ---
def norm(series):
    return (series - series.min()) / (series.max() - series.min())

# --- Word-level Plot Data ---
word_stats_emot = word_stats_emot.copy()
word_stats_emot = word_stats_emot[(word_stats_emot['avg_frequency'] > 0) & (word_stats_emot['need_odds'] > 0)]

word_stats_emot['log_freq'] = np.log(word_stats_emot['avg_frequency'])
word_stats_emot['log_need'] = np.log(word_stats_emot['need_odds'])
word_stats_emot['arousal_n'] = norm(word_stats_emot['arousal'])

# --- Frequency-level Plot Data ---
freq_stats_emot = freq_stats_emot.copy()
freq_stats_emot = freq_stats_emot[(freq_stats_emot['frequency'] > 0) & (freq_stats_emot['need_odds'] > 0)]

freq_stats_emot['log_freq'] = np.log(freq_stats_emot['frequency'])
freq_stats_emot['log_need'] = np.log(freq_stats_emot['need_odds'])
freq_stats_emot['arousal_n'] = norm(freq_stats_emot['mean_arousal'])

print(freq_stats_emot)
# # --------------------------------------------------  PLOTS  ------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# vmin = min(word_stats_emot['valence'].min(), freq_stats_emot['mean_valence'].min())
# vmax = max(word_stats_emot['valence'].max(), freq_stats_emot['mean_valence'].max())

vmin = 1
vmax = 9

# (0,0) WORD LEVEL
sc1 = axes[0].scatter(
    word_stats_emot['log_freq'], word_stats_emot['log_need'],
    s = 20 + 380 * word_stats_emot['arousal_n'],
    c = word_stats_emot['valence'],
    cmap = 'viridis',
    alpha = 0.25 + 0.75 * word_stats_emot['arousal_n'],
    edgecolor='k', linewidths=0,
    vmin=vmin, vmax=vmax
)
axes[0].set_title('Group by Word: Arousal→Size/Opacity, Valence→Colour')
axes[0].set_ylabel('Log Need Odds')
fig.colorbar(sc1, ax=axes[0], label='Valence')

# (1,0) FREQ LEVEL
sc2 = axes[1].scatter(
    freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
    s = 20 + 380 * freq_stats_emot['arousal_n'],
    c = freq_stats_emot['mean_valence'],
    cmap = 'viridis',
    alpha = 0.25 + 0.75 * freq_stats_emot['arousal_n'],
    edgecolor='k', linewidths=0,
    vmin=vmin, vmax=vmax
)
axes[1].set_title('Group by Freq: Arousal→Size/Opacity, Valence→Colour')
axes[1].set_xlabel('Log Frequency')
axes[1].set_ylabel('Log Need Odds')
fig.colorbar(sc2, ax=axes[1], label='Mean Valence')

# Add gridlines
for ax in axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# plt.tight_layout()
# plt.show()


# -- Valence and Arousal separated -----------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

# Word-level valence
sc0 = axes[0, 0].scatter(word_stats_emot['log_freq'], word_stats_emot['log_need'],
                         c=word_stats_emot['valence'], cmap='coolwarm', alpha=0.7,
                         vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Word: Valence ")
fig.colorbar(sc0, ax=axes[0, 0], label="Valence")

# Word-level arousal
sc1 = axes[1, 0].scatter(word_stats_emot['log_freq'], word_stats_emot['log_need'],
                         c=word_stats_emot['arousal'], cmap='coolwarm', alpha=0.7, vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Word: Arousal ")
fig.colorbar(sc1, ax=axes[1, 0], label="Arousal")

# Frequency-bucket valence
sc2 = axes[0, 1].scatter(freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
                         c=freq_stats_emot['mean_valence'], cmap='coolwarm', alpha=0.8,
                         vmin=vmin, vmax=vmax)
axes[0, 1].set_title("Freq: Valence ")
fig.colorbar(sc2, ax=axes[0, 1], label="Mean Valence")

# Frequency-bucket arousal
sc3 = axes[1, 1].scatter(freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
                         c=freq_stats_emot['mean_arousal'], cmap='coolwarm', alpha=0.8, vmin=vmin, vmax=vmax)
axes[1, 1].set_title("Freq: Arousal ")
fig.colorbar(sc3, ax=axes[1, 1], label="Mean Arousal")

for ax in axes[1, :]:
    ax.set_xlabel("log Frequency")
for ax in axes[:, 0]:
    ax.set_ylabel("log Need Odds")

for row in axes:
    for ax in row:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


# -- Split valence into 3 groups ---------------------------------------------------------------
# word_stats_emot['valence_group'] = pd.cut(word_stats_emot['valence'], bins=3, labels=["Low", "Mid", "High"])
# freq_stats_emot['valence_group'] = pd.cut(freq_stats_emot['mean_valence'], bins=3, labels=["Low", "Mid", "High"])

# fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# for col, group in enumerate(["Low", "Mid", "High"]):
#     subset = word_stats_emot[word_stats_emot['valence_group'] == group]
#     sc = axes[0, col].scatter(subset['log_freq'], subset['log_need'],
#                               c=subset['arousal'], cmap='coolwarm', alpha=0.8, vmin=vmin, vmax=vmax)
#     axes[0, col].set_title(f"Word: {group} Valence")
#     axes[0, col].set_xlabel("log Frequency")
#     axes[0, col].set_ylabel("log Need Odds")
# fig.colorbar(sc, ax=axes[0, 2], label="Arousal")

# for col, group in enumerate(["Low", "Mid", "High"]):
#     subset = freq_stats_emot[freq_stats_emot['valence_group'] == group]
#     sc = axes[1, col].scatter(subset['log_freq'], subset['log_need'],
#                               c=subset['mean_arousal'], cmap='coolwarm', alpha=0.8, vmin=vmin, vmax=vmax)
#     axes[1, col].set_title(f"Freq: {group} Valence")
#     axes[1, col].set_xlabel("log Frequency ")
#     axes[1, col].set_ylabel("log Need Odds")
# fig.colorbar(sc, ax=axes[1, 2], label="Mean Arousal")

# for row in axes:
#     for ax in row:
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()
