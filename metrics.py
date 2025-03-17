import pandas as pd
from collections import Counter
from data import text_df, WORD_LEN

all_words = ["SKIP"] * WORD_LEN 
for _, row in text_df.iterrows():
    word = row['WORD']
    for pos in row['POSITIONS']:
        all_words[pos] = word

window_size = 100
results = []

for i in range(len(all_words) - window_size - 1):
    window = all_words[i:i+window_size]
    next_word = all_words[i+window_size]
    
    # Skip windows with too many placeholder words or if next word is a placeholder
    if window.count("SKIP") > window_size/2 or next_word == "SKIP":
        continue
        
    window_words = [w for w in window if w != "SKIP"] # excluding SKIP placeholders
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

word_stats = recall_df.groupby('word')['is_recalled'].agg(
    recall_probability=('mean'),
    count=('count'), #number of windown with the word
    recalls=('sum') #number of times the word was the next word
).reset_index().sort_values('recall_probability', ascending=False)

#avg frequency over the windows
avg_freq = recall_df.groupby('word')['frequency'].mean().reset_index()
word_stats = word_stats.merge(avg_freq, on='word')

print(recall_df)
print(word_stats)