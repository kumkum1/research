import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from metrics import word_stats, freq_stats
from scipy.stats import linregress

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# --- 1. Word Frequency vs Recall Probability ---
sns.scatterplot(
    x='avg_frequency', 
    y='recall_probability', 
    data=word_stats,
    ax=axes[0, 0]
)

axes[0, 0].set_title(f'Word Frequency vs Recall Probability(group by word)')
axes[0, 0].set_xlabel('Average Frequency')
axes[0, 0].set_ylabel('Recall Probability')
axes[0, 0].grid(True)

# --- 2. Log Transformation for Word Stats ---
word_stats_log = word_stats[(word_stats['avg_frequency'] > 0) & (word_stats['recall_probability'] > 0)].copy()
word_stats_log['log_avg_frequency'] = np.log10(word_stats_log['avg_frequency'])
word_stats_log['log_recall_probability'] = np.log10(word_stats_log['recall_probability'])

sns.scatterplot(
    x='log_avg_frequency', 
    y='log_recall_probability', 
    data=word_stats_log,
    ax=axes[0, 1]
)
axes[0, 1].set_title(f'Log Transformation: Avg Frequency vs Recall Probability')
axes[0, 1].set_xlabel('Log Avg Frequency')
axes[0, 1].set_ylabel('Log Recall Probability')
axes[0, 1].grid(True)

# --- 3. Linear Regression on Log-Transformed Data ---
slope_fw, intercept_fw, r_value_fw, _, _ = linregress(
    word_stats_log['log_avg_frequency'], 
    word_stats_log['log_recall_probability']
)
word_stats_log['predicted'] = slope_fw * word_stats_log['log_avg_frequency'] + intercept_fw
word_stats_log['residual'] = np.abs(word_stats_log['log_recall_probability'] - word_stats_log['predicted'])
threshold = 0.2
close_points = word_stats_log[word_stats_log['residual'] < threshold].copy()
close_points = close_points.sort_values('log_avg_frequency')


axes[0, 2].plot(close_points['log_avg_frequency'], close_points['log_recall_probability'], color='blue', linestyle='--')
x_vals = np.linspace(word_stats_log['log_avg_frequency'].min(), word_stats_log['log_avg_frequency'].max(), 100)
axes[0, 2].plot(x_vals, slope_fw * x_vals + intercept_fw, color='red', linestyle='-', label='Regression Line')

sorted_log_data = word_stats_log.sort_values('log_avg_frequency')
sns.scatterplot(
    x='log_avg_frequency', 
    y='log_recall_probability',
    data=sorted_log_data,
    ax=axes[0, 2],
    alpha=0.5,
    label=f'y = {slope_fw:.2f}x + {intercept_fw:.2f}\nR² = {r_value_fw**2:.3f}'
)
axes[0, 2].set_title('Log-Log: Avg Frequency vs Recall Probability')
axes[0, 2].set_xlabel('Log Avg Frequency')
axes[0, 2].set_ylabel('Log Recall Probability')
axes[0, 2].grid(True)

# --- 4. Frequency Stats Analysis ---
sns.scatterplot(
    x='frequency', 
    y='recall_probability', 
    data=freq_stats, 
    ax=axes[1, 0]
)
axes[1, 0].set_title('Word Frequency vs Recall Probability (group by frequency)')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Recall Probability')
axes[1, 0].grid(True)

# --- 5. Log Transformation for Frequency Stats ---
freq_stats_log = freq_stats[freq_stats['frequency'] > 0].copy()        
freq_stats_log['log_frequency'] = np.log10(freq_stats_log['frequency'])
freq_stats_log['log_recall_probability'] = np.log10(freq_stats_log['recall_probability'])

sns.scatterplot(
    x='log_frequency', 
    y='log_recall_probability', 
    data=freq_stats_log, 
    ax=axes[1, 1]
)
axes[1, 1].set_title('Log Transformation: Frequency vs Recall Probability')
axes[1, 1].set_xlabel('Log Frequency')
axes[1, 1].set_ylabel('Log Recall Probability')
axes[1, 1].grid(True)

# --- 6. Linear Regression for Log-Transformed Frequency Stats ---
log_y_data = freq_stats_log[freq_stats_log['recall_probability'] > 0].copy()
log_y_data['log_recall_probability'] = np.log10(log_y_data['recall_probability'])

slope_fs, intercept_fs, r_value_fs, _, _ = linregress(
    log_y_data['log_frequency'], 
    log_y_data['log_recall_probability']
)
freq_stats_log['predicted'] = slope_fs * freq_stats_log['log_frequency'] + intercept_fs
freq_stats_log['residual'] = np.abs(freq_stats_log['log_recall_probability'] - freq_stats_log['predicted'])
threshold = 0.2
freq_close_points = freq_stats_log[freq_stats_log['residual'] < threshold].copy()
freq_close_points = freq_close_points.sort_values('log_frequency')

sorted_log_freq_data = log_y_data.sort_values('log_frequency')
sns.scatterplot(
    x='log_frequency', 
    y='log_recall_probability',
    data=sorted_log_freq_data,
    ax=axes[1, 2],
    alpha=0.5
)

x_values = sorted_log_freq_data['log_frequency']
reg_line_fs = slope_fs * x_values + intercept_fs
axes[1, 2].plot(
    x_values, 
    reg_line_fs, 
    color='red', 
    linestyle='-', 
    label=f'y = {slope_fs:.2f}x + {intercept_fs:.2f}\nR² = {r_value_fs**2:.3f}'
)
axes[1, 2].plot(freq_close_points['log_frequency'], freq_close_points['log_recall_probability'], color='blue', linestyle='--')

axes[1, 2].legend()
axes[1, 2].set_title('Log-Log: Frequency vs Recall Probability')
axes[1, 2].set_xlabel('Log Frequency')
axes[1, 2].set_ylabel('Log Recall Probability')
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()