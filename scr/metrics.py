import pandas as pd
from collections import Counter
from data import text_df, WORD_LEN
from plot import plot_stats

all_words = ["skip"] * WORD_LEN 
for _, row in text_df.iterrows():
    word = row['word']
    for pos in row['positions']:
        all_words[pos] = word

window_size = 400 
results = []

for i in range(len(all_words) - window_size - 1):
    window = all_words[i:i+window_size]
    next_word = all_words[i+window_size]
    
    # Skip windows with too many placeholder words or if next word is a placeholder
    if window.count("skip") > window_size/2 or next_word == "skip":
        continue
        
    window_words = [w for w in window if w != "skip"] # excluding SKIP placeholders
    
    word_counts = Counter(window_words)
         
    for word, freq in word_counts.items():
        is_recalled = 1 if word == next_word else 0
        results.append({
            'window_start': i,
            'window_end': i+window_size-1,
            'word': word,
            'frequency': freq,
            'is_recalled': is_recalled
        })
recall_df = pd.DataFrame(results)

#group by word
word_stats = recall_df.groupby('word')['is_recalled'].agg(
    recall_probability=('mean'),
    count=('count'),
    recalls=('sum') 
).reset_index().sort_values('recall_probability', ascending=False)
avg_freq = recall_df.groupby('word')['frequency'].mean().reset_index()
avg_freq = avg_freq.rename(columns={'frequency': 'avg_frequency'})
word_stats = word_stats.merge(avg_freq, on='word')
word_stats['need_odds'] = word_stats['recall_probability'].transform(lambda x: x/(1-x))

freq_stats = recall_df.groupby('frequency').agg(
    total=('is_recalled', 'count'),
    recalls=('is_recalled', 'sum'),
).reset_index()
freq_stats['recall_probability'] = freq_stats['recalls'] / freq_stats['total']
freq_stats['need_odds'] = freq_stats['recall_probability'].transform(lambda x: x/(1-x))
freq_stats['unique_words'] = recall_df.groupby('frequency')['word'].unique().reset_index(drop=True)

# print(freq_stats)
plot_stats(word_stats, freq_stats)

# Save the results to CSV files
word_stats.to_csv('data/processed/word_stats.csv', index=False)
freq_stats.to_csv('data/processed/freq_stats.csv', index=False)
recall_df.to_csv('data/processed/recall_df.csv', index=False)