import pandas as pd
from collections import Counter
from data import text_df, WORD_LEN
from plot import plot_stats

all_words = ["SKIP"] * WORD_LEN 
for _, row in text_df.iterrows():
    word = row['WORD']
    for pos in row['POSITIONS']:
        all_words[pos] = word

window_size = 200 #200 seems to work better than a 100 and 300 window size
results = []

for i in range(len(all_words) - window_size - 1):
    window = all_words[i:i+window_size]
    next_word = all_words[i+window_size]
    
    # Skip windows with too many placeholder words or if next word is a placeholder
    if window.count("SKIP") > window_size/2 or next_word == "SKIP":
        continue
        
    window_words = [w for w in window if w != "SKIP"] # excluding SKIP placeholders
    
    # word_counts = Counter(window_words)
    # for word in window_words:
    #     if word == next_word:
    #         word_counts[word] 

    freq = window_words.count(next_word)
    results.append({
        'window_start': i,
        'window_end': i+window_size-1,
        'word': next_word,
        'frequency': freq,
        'is_recalled': 1
    })
         
    # for word, freq in word_counts.items():
    #     is_recalled = 1 if word == next_word else 0
    #     results.append({
    #         'window_start': i,
    #         'window_end': i+window_size-1,
    #         'word': word,
    #         'frequency': freq,
    #         'is_recalled': is_recalled
    #     })
    
recall_df = pd.DataFrame(results)
print(recall_df)

#group by word
word_stats = recall_df.groupby('word')[['is_recalled', 'frequency']].agg(
    ('sum')
).reset_index()
print(word_stats)
word_stats.rename(columns={'is_recalled': 'recalls'}, inplace=True)
word_stats['avg_frequency'] = word_stats['frequency']/(len(all_words)-window_size-1)
word_stats['recall_probability'] = word_stats['recalls']/(len(all_words)-window_size-1)
print( (1-word_stats['recall_probability']))
word_stats['need_odds'] = word_stats['recall_probability'].transform(lambda x: x/(1-x))
# word_stats = word_stats.assign(avg_frequency=avg_freq)
# word_stats = word_stats.rename(columns={
#     'recall_probability': 'temp',
#     'avg_frequency': 'recall_probability',
#     "recalls": 'avg_frequency'
# })
# word_stats = word_stats.rename(columns={'temp': 'avg_frequency'})


# avg_freq = recall_df.groupby('word')['recall'].mean().reset_index()
# word_stats = word_stats.merge(avg_freq, on='word')
# word_stats = word_stats.rename(columns={'frequency': 'avg_frequency'})

#group by frequency
# freq_stats = recall_df.groupby('frequency')[['is_recalled']].agg(
#     ('sum')
# ).reset_index()



print(recall_df)
print(word_stats)
# print(freq_stats.sort_values('frequency', ascending=False))


plot_stats(word_stats)