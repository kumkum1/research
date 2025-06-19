import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.cm import ScalarMappable
from metrics import word_stats

# Load and clean emotion data
emot_df = pd.read_csv("emot_28724.csv")
emot_df.columns = emot_df.columns.str.strip().str.lower()
emot_df = emot_df[['word', 'valence', 'arousal']]

# Filter and merge
word_stats = word_stats[['word', 'avg_frequency', 'need_odds']]
word_stats_emotion = word_stats.merge(emot_df, on='word', how='inner')
word_stats_emotion.dropna(subset=['valence', 'arousal'], inplace=True)

# Normalize valence and arousal
word_stats_emotion['valence_norm'] = (word_stats_emotion['valence'] - word_stats_emotion['valence'].min()) / \
                                     (word_stats_emotion['valence'].max() - word_stats_emotion['valence'].min())
word_stats_emotion['size'] = 30 + 70 * (
    (word_stats_emotion['arousal'] - word_stats_emotion['arousal'].min()) /
    (word_stats_emotion['arousal'].max() - word_stats_emotion['arousal'].min())
)
hsv_colors = plt.cm.hsv(word_stats_emotion['valence_norm'])

# Add log values
word_stats_emotion['log_freq'] = np.log(word_stats_emotion['avg_frequency'])
word_stats_emotion['log_need'] = np.log(word_stats_emotion['need_odds'])

# Plotting all three
fig, axes = plt.subplots(1, 2, figsize=(28, 8))

# Viridis Plot
viridis = axes[0].scatter(
    word_stats_emotion['avg_frequency'],
    word_stats_emotion['need_odds'],
    c=word_stats_emotion['valence'],
    s=word_stats_emotion['size'],
    cmap='viridis', alpha=0.8, edgecolor='k'
)
axes[0].set_title('Viridis: Valence | Size: Arousal')
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Need Odds')
fig.colorbar(viridis, ax=axes[0], label='Valence')

# HSV Hue Plot (Linear)
hsv1 = axes[1].scatter(
    word_stats_emotion['avg_frequency'],
    word_stats_emotion['need_odds'],
    c=hsv_colors,
    s=word_stats_emotion['size'],
    alpha=0.8, edgecolor='k'
)
axes[1].set_title('HSV Hue: Valence | Size: Arousal')
axes[1].set_xlabel('Frequency')
fig.colorbar(
    ScalarMappable(cmap='hsv', norm=plt.Normalize(
        vmin=word_stats_emotion['valence'].min(),
        vmax=word_stats_emotion['valence'].max())),
    ax=axes[1], label='Valence (Hue)'
)

plt.tight_layout()
plt.show()
