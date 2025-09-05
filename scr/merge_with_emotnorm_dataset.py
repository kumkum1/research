import pandas as pd
import numpy as np         

word_stats = pd.read_csv('data/processed/word_stats.csv') # 12132
print(word_stats.shape)
word_stats = word_stats[word_stats['need_odds'] > 0] #5529
print(word_stats.shape)
recall_df = pd.read_csv('data/processed/recall_df.csv')
emot_df = pd.read_csv('data/raw/emot_28724.csv') #28724
print(emot_df.shape)


emot_df.rename(columns=lambda x: x.lower().strip(), inplace=True)
emot_df = emot_df[['word', 'valence', 'arousal']]
df = pd.merge(word_stats, emot_df, on='word', how='left')
df.to_csv('data/merged/combined_with_emotion.csv', index=False) #5529
print(df.shape)

# word statistics merged with emotion
df[df['valence'].isna() | df['arousal'].isna()] #2362 missing both valence and arousal
word_stats_emot = df.dropna(subset=['valence', 'arousal']) #3167
print(word_stats_emot.shape)

# Normalization 
def norm(series):
    return (series - series.min()) / (series.max() - series.min())

# Word Plot Data 
word_stats_emot = ( #3167
    word_stats_emot
    .assign(
        log_freq=lambda d: np.log(d['avg_frequency']),
        log_need=lambda d: np.log(d['need_odds']),
        arousal_n=lambda d: norm(d['arousal'])
    )
) 
print(word_stats_emot.shape)

word_stats_emot.to_csv("data/merged/word_stats_emot.csv", index=False)

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

#  Frequency Plot Data 
freq_stats_emot = (
    freq_stats_emot
    .assign(
        log_freq=lambda d: np.log(d['frequency']),
        log_need=lambda d: np.log(d['need_odds']),
        arousal_n=lambda d: norm(d['mean_arousal'])
    )
)

freq_stats_emot.to_csv("data/merged/freq_stats_emot.csv", index=False)

