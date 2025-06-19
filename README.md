# Word Frequency and Recall Analysis

## Research Context

The analysis follows Anderson & Schooler's (1991) methodology for examining word usage patterns in news headlines over time. In their study, they observed that if a word (e.g., "Reagan") appears 52 times in a 100-day window, they could analyze its probability of appearing on the 101st day. 

In our implementation:
- We use a sliding window of 200 words (analogous to the 100-day window in Anderson & Schooler's news headline study)
- We analyze how word frequency within this window predicts its appearance in the next position
- We calculate "need odds" (recall probability) to estimate the likelihood of a word appearing based on its frequency

## Structure

- `data.py`: Handles data loading and preprocessing from the Reuters corpus. Loads and structures the raw data into `text_df`, including word positions and metadata.
- `metrics.py`: Implements the sliding window analysis and calculates recall probabilities
- `plot.py`: Generates visualizations of the analysis results

## Output

The program generates a series of plots showing:
1. Need odds vs. frequency (by word)
2. Log-transformed analysis of need odds vs. frequency
3. Linear regression analysis
4. Frequency-based statistics
5. Log-transformed frequency analysis
6. Regression analysis of log-transformed data


## Hypotheses
1. Frequency - Valence 
- Words that occur more frequently will have a higher(positive) valence
2. Frequency - Arousal
- More frequent words would have lower arousal
3. Valence - Arousal
- Words with extreme valence (very positive or very negative) would also have higher arousal 
4. Frequency - Dominance
- 