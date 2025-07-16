import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from metrics import word_stats, freq_stats          # <-- both aggregations

# --------------------------------------------------  DATA  -------------------
emot_df = (
    pd.read_csv("emot_28724.csv")
      .pipe(lambda d: d.rename(columns=str.strip).rename(columns=str.lower))
      [['word', 'valence', 'arousal']]
)

# ---------- WORD-level frame --------------------------------------------------
ws = word_stats[['word', 'avg_frequency', 'need_odds']]
word_df = (ws
           .merge(emot_df, on='word', how='inner')
           .dropna(subset=['valence', 'arousal'])
           .assign(log_freq = lambda d: np.log(d['avg_frequency']),
                   log_need = lambda d: np.log(d['need_odds']))
)

# ---------- FREQ-bucket frame -------------------------------------------------
# 1) attach emotion to each word & bucket by integer frequency
bucketed = (ws
            .merge(emot_df, on='word', how='inner')
            .dropna(subset=['valence', 'arousal'])
            .assign(freq_bucket = lambda d: d['avg_frequency'].round().astype(int))
)

print(bucketed)

# 2) average emotion within each bucket, then join to recall stats
emot_by_freq = (bucketed.groupby('freq_bucket')
                          .agg(mean_valence=('valence', 'mean'),
                               mean_arousal=('arousal', 'mean'))
                          .reset_index())

print(emot_by_freq)

freq_df = (freq_stats.reset_index()
           .rename(columns={'frequency':'freq_bucket'})
           .merge(emot_by_freq, on='freq_bucket', how='inner')
           .assign(log_freq = lambda d: np.log(d['freq_bucket']),
                   log_need = lambda d: np.log(d['need_odds']))
)

# ---------- normalisations for point size / alpha ---------------------------
def norm(series):           # min-max 0–1 helper
    return (series - series.min()) / (series.max() - series.min())

word_arousal_n  = norm(word_df['arousal'])
word_valence_n  = norm(word_df['valence'])
freq_arousal_n  = norm(freq_df['mean_arousal'])
freq_valence_n  = norm(freq_df['mean_valence'])

# --------------------------------------------------  PLOTS  ------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# -- (0,0) WORD  ·  Arousal→size/opacity , Valence→colour ─────────────────────
sc = axes[0].scatter(
    word_df['log_freq'], word_df['log_need'],
    s = 20 + 380*word_arousal_n,
    c = word_df['valence'],
    cmap='viridis',
    alpha = 0.25 + 0.75*word_arousal_n,
    edgecolor='k', linewidths=0
)
axes[0].set_title('Group by Word: Arousal→Size/Opacity, Valence→Colour')
axes[0].set_ylabel('Log Need Odds')
fig.colorbar(sc, ax=axes[0], label='Valence')

# -- (1,0) FREQ  ·  Arousal→size/opacity , Valence→colour --------------------------
sc = axes[1].scatter(
    freq_df['log_freq'], freq_df['log_need'],
    s = 20 + 380*freq_arousal_n,
    c = freq_df['mean_valence'],
    cmap='viridis',
    alpha = 0.25 + 0.75*freq_arousal_n,
    edgecolor='k', linewidths=0
)
axes[1].set_title('Group by Freq: Arousal→Size/Opacity, Valence→Colour')
axes[1].set_xlabel('Log Frequency')
axes[1].set_ylabel('Log Need Odds')
fig.colorbar(sc, ax=axes[1], label='Mean Valence')

for ax in axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


# plt.xlim(0.0, 3.0)  # e.g., plt.xlim(0, 100)
# plt.ylim(-7, 0)  # e.g., plt.ylim(0, 1)

print(freq_df)

# -- Valence and Arousal seperated -----------------------------------------
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

# # Word-level valence
# sc0 = axes[0, 0].scatter(word_df['log_freq'], word_df['log_need'],
#                          c=word_df['valence'], cmap='coolwarm', alpha=0.7)
# axes[0, 0].set_title("Word: Valence ")
# fig.colorbar(sc0, ax=axes[0, 0], label="Valence")

# # Word-level arousal
# sc1 = axes[1, 0].scatter(word_df['log_freq'], word_df['log_need'],
#                          c=word_df['arousal'], cmap='coolwarm', alpha=0.7)
# axes[1, 0].set_title("Word: Arousal ")
# fig.colorbar(sc1, ax=axes[1, 0], label="Arousal")

# # Frequency-bucket valence
# sc2 = axes[0, 1].scatter(freq_df['log_freq'], freq_df['log_need'],
#                          c=freq_df['mean_valence'], cmap='coolwarm', alpha=0.8)
# axes[0, 1].set_title("Freqt: Valence ")
# fig.colorbar(sc2, ax=axes[0, 1], label="Mean Valence")

# # Frequency-bucket arousal
# sc3 = axes[1, 1].scatter(freq_df['log_freq'], freq_df['log_need'],
#                          c=freq_df['mean_arousal'], cmap='coolwarm', alpha=0.8)
# axes[1, 1].set_title("Freq: Arousal ")
# fig.colorbar(sc3, ax=axes[1, 1], label="Mean Arousal")

# for ax in axes[1, :]:
#     ax.set_xlabel("log Frequency")
# for ax in axes[:, 0]:
#     ax.set_ylabel("log Need Odds")

# for row in axes:
#     for ax in row:
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


# -- Split valence into 3 groups ---------------------------------------------------------------

# word_df['valence_group'] = pd.cut(word_df['valence'], bins=3, labels=["Low", "Mid", "High"])
# freq_df['valence_group'] = pd.cut(freq_df['mean_valence'], bins=3, labels=["Low", "Mid", "High"])

# fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# for col, group in enumerate(["Low", "Mid", "High"]):
#     subset = word_df[word_df['valence_group'] == group]
#     sc = axes[0, col].scatter(subset['log_freq'], subset['log_need'],
#                               c=subset['arousal'], cmap='coolwarm', alpha=0.8)
#     axes[0, col].set_title(f"Word: {group} Valence")
#     axes[0, col].set_xlabel("log Frequency")
#     axes[0, col].set_ylabel("log Need Odds")
# fig.colorbar(sc2, ax=axes[0, 2], label="Arousal")

# for col, group in enumerate(["Low", "Mid", "High"]):
#     subset = freq_df[freq_df['valence_group'] == group]
#     sc = axes[1, col].scatter(subset['log_freq'], subset['log_need'],
#                               c=subset['mean_arousal'], cmap='coolwarm', alpha=0.8)
#     axes[1, col].set_title(f"Freq: {group} Valence")
#     axes[1, col].set_xlabel("log Frequency ")
#     axes[1, col].set_ylabel("log Need Odds")
# fig.colorbar(sc2, ax=axes[1, 2], label="Mean Arousal")

# for row in axes:
#     for ax in row:
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


# plt.tight_layout()
# plt.show()
