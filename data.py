import nltk
from nltk.corpus import stopwords, reuters
from collections import Counter, defaultdict
import pandas as pd

nltk.download('reuters')
nltk.download('stopwords')

# Load words from the first Reuters file
words = reuters.words(reuters.fileids()) [:1000]
words = [word.lower() for word in words] 

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

word_freq = Counter(filtered_words)

words_positions = defaultdict(list)
for index, word in enumerate(filtered_words):
    if word in filtered_words:
        words_positions[word].append(index)

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