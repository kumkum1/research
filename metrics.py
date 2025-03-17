import pandas as pd
from collections import Counter
from data import text_df

records = []
words = text_df['WORD'].tolist()
window_size = 100

    
for start_idx in range(len(words) - window_size):
    end_idx = start_idx + window_size - 1  
    
    window = words[start_idx : start_idx + window_size]
    next_word = words[start_idx + window_size]
    
    freq_counter = Counter(window)
    
    for w, freq in freq_counter.items():
        next_word_flag = 1 if (w == next_word) else 0
        records.append({
            'Window_Start': start_idx,
            'Window_End': end_idx,
            'Word': w,
            'Frequency': freq,
            'Next_Word': next_word,
            'Next_Word_Flag': next_word_flag
        })
    
df = pd.DataFrame.from_records(records)

print(df)
