import nltk
from nltk.corpus import stopwords, reuters
from collections import Counter, defaultdict
import pandas as pd

nltk.download('reuters', quiet=True)
nltk.download('stopwords', quiet=True)

# Load words from the first Reuters file
WORD_LEN = 1000
words = [word.lower() for word in reuters.words(reuters.fileids())[:WORD_LEN]]

# filter
stop_words = set(stopwords.words('english'))
filtered_words = []

words_positions = defaultdict(list)
for idx, word in enumerate(words):
    if word.isalpha() and word not in stop_words:
        filtered_words.append(word)
        words_positions[word].append(idx)

word_freq = Counter(filtered_words)

#display
text_data = []
for word, positions in words_positions.items():
    text_data.append({
        'WORD': word,
        'FREQUENCY': word_freq[word],
        'POSITIONS': positions
    })
text_df = pd.DataFrame(text_data).sort_values(by='FREQUENCY', ascending=False)

# print(text_df)
# text_df.to_csv('text_data.csv', index=False)