import pandas as pd
from metrics import word_stats, recall_df
import numpy as np         

emot_df = pd.read_csv('emot_28724.csv')
emot_df.rename(columns=lambda x: x.lower().strip(), inplace=True)
emot_df = emot_df[['word', 'valence', 'arousal']]
df = pd.merge(word_stats, emot_df, on='word', how='left')
df.to_csv('combined_with_emotion.csv', index=False)

# word statistics merged with emotion
word_stats_emot = df.dropna(subset=['valence', 'arousal'])

# frequency statistics merged with emotion
recalled_with_emotion = (
    recall_df.merge(emot_df, on='word', how='left')
)

freq_stats_emot = recalled_with_emotion.groupby('frequency').agg(
    total=('is_recalled', 'count'),
    recalls=('is_recalled', 'sum'),
    mean_valence=('valence', 'mean'),
    mean_arousal=('arousal', 'mean')
).reset_index()

freq_stats_emot['recall_probability'] = freq_stats_emot['recalls'] / freq_stats_emot['total']
freq_stats_emot['need_odds'] = freq_stats_emot['recall_probability'].transform(lambda x: x/(1-x))
freq_stats_emot.dropna(subset=['mean_valence', 'mean_arousal'], inplace=True)

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

# print(freq_stats_emot)