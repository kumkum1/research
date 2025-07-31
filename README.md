# Word Frequency and Recall Analysis with Emotional Norms

This research project investigates the relationship between word frequency, emotional characteristics (valence and arousal), and memory recall probability. This research builds upon **Anderson & Schooler's (1991) need probability model**, which examines how word frequency in a context window predicts its appearance in subsequent positions. Our implementation adapts their methodology to analyze emotional dimensions alongside frequency effects.

## Implemntation

#### 1. Corpus Preparation (`data.py`)
- **Source**: Reuters corpus (first 50,000 words)
- **Preprocessing**:
  - Tokenization and Stopword removal 
  - Acronym expansion 
  - Position tracking for each word

#### 2. Sliding Window Analysis (`metrics.py`)
- **Window Size**: 400 words (analogous to Anderson & Schooler's 100-day window)
- **Step Size**: 1 word (sliding window approach)
- **Recall Calculation**: For each word in the window, determine if it appears as the next word
- **Statistics Computed**:
  - Recall probability: P(recall|frequency)
  - Need odds: recall_probability / (1 - recall_probability)
  - Frequency statistics and word-level metrics

#### 3. Emotional Integration (`merge_with_emotnorm_dataset.py`)
- **Emotional Norms**: Merges with `emot_28724.csv` containing valence and arousal ratings
- **Data Cleaning**: Handles missing emotional data
- **Feature Engineering**: Creates log-transformed frequency and need odds

## Statistical Analysis

### Regression Models (`regression.py`)

1. **Model 1**: Main effects only
   ```
   log_need ~ log_freq + valence_dm + arousal_dm
   ```

2. **Model 2**: Frequency × Valence interaction
   ```
   log_need ~ log_freq * valence_dm
   ```

3. **Model 3**: Frequency × Arousal interaction
   ```
   log_need ~ log_freq * arousal_dm
   ```

4. **Model 4**: Three-way interaction
   ```
   log_need ~ log_freq * valence_dm * arousal_dm
   ```

5. **Model 5**: Combined interaction effects
   ```
   log_need ~ log_freq * valence_dm + log_freq * arousal_dm
   ```

6. **Model 6**: Categorical analysis
   ```
   log_need ~ log_freq * C(valence_group) * C(arousal_group)
   ```

## Visualizations 

1. **Basic Analysis Plots** (`basic_analysis_plots.png`)
   - Need odds vs. frequency (by word and frequency)
   - Log-transformed analyses

2. **Emotional Analysis** (`analysis_with_emotional_norm.png`)
   - Frequency vs. recall with arousal (size/opacity) and valence (color)
   - Separate analyses for word-level and frequency-level data

3. **Split Analysis** (`word_freq_split_valence_arousal.png`)
   - Individual plots for valence and arousal effects
   - Color-coded emotional dimensions

4. **3×3 Grid Analysis** (`split_valence_arousal_9plots.png`)
   - Detailed breakdown by valence and arousal groups
   - Individual regression lines for each combination

5. **Combined Summary** (`combined_3split_summary_plot.png`)
   - Comprehensive view of all emotional dimensions
   - Color-coded valence groups with size-coded arousal

## Results

## References

- Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. *Psychological Science*, 2(6), 396-408.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45(4), 1191-1207.




