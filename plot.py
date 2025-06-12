import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

def plot_stats(word_stats, freq_stats):
    _, axes = plt.subplots(2, 3, figsize=(18, 8))

    # --- Word Frequency vs Recall Probability ---
    sns.scatterplot(
        x='avg_frequency', 
        y='need_odds', 
        data=word_stats,
        ax=axes[0, 0],
        alpha=0.5,
        linewidth=0
    )

    axes[0, 0].set_title(f'Need Odds vs Frequency (by Word)')
    axes[0, 0].set_xlabel('Avg Frequency')
    axes[0, 0].set_ylabel('Need Odds')
    axes[0, 0].grid(True)

    # --- Log Transformation for Word Stats ---
    word_stats_log = word_stats[(word_stats['avg_frequency'] > 0) & (word_stats['need_odds'] > 0)].copy()
    word_stats_log['log_avg_frequency'] = np.log(word_stats_log['avg_frequency'])
    word_stats_log['log_need_odds'] = np.log(word_stats_log['need_odds'])

    sns.scatterplot(
        x='log_avg_frequency', 
        y='log_need_odds', 
        data=word_stats_log,
        ax=axes[0, 1],
        alpha=0.5,
        linewidth=0
    )
    axes[0, 1].set_title(f'Log Transformation: Need Odds vs Avg Frequency')
    axes[0, 1].set_xlabel('Log Avg Frequency')
    axes[0, 1].set_ylabel('Log Need Odds')
    axes[0, 1].grid(True)

    # --- Linear Regression on Log-Transformed Data ---
    slope_fw, intercept_fw, r_value_fw, _, _ = linregress(
        word_stats_log['log_avg_frequency'], 
        word_stats_log['log_need_odds']
    )
    x_vals = word_stats_log['log_avg_frequency']
    axes[0, 2].plot(x_vals, slope_fw * x_vals + intercept_fw, color='red', linestyle='-', label='Regression Line')

    sns.scatterplot(
        x='log_avg_frequency', 
        y='log_need_odds',
        data=word_stats_log.sort_values('log_avg_frequency'),
        ax=axes[0, 2],
        alpha=0.5,
        linewidth=0,
        label=f'y = {slope_fw:.2f}x + {intercept_fw:.2f}\nR² = {r_value_fw**2:.3f}'
    )
    axes[0, 2].set_title('Log Need Odds vs Log Avg Frequency (Regression Line)')
    axes[0, 2].set_xlabel('Log Avg Frequency')
    axes[0, 2].set_ylabel('Log Need Odds')
    axes[0, 2].grid(True)


    # --- Frequency Stats Analysis ---
    sns.scatterplot(
        x='frequency', 
        y='need_odds', 
        data=freq_stats, 
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Need odds vs Frequency (by frequency)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Need Odds')
    axes[1, 0].grid(True)

    # --- Log Transformation for Frequency Stats ---
    freq_stats_log = freq_stats[freq_stats['frequency'] > 0].copy()        
    freq_stats_log['log_frequency'] = np.log(freq_stats_log['frequency'])
    freq_stats_log['log_need_odds'] = np.log(freq_stats_log['need_odds'])

    sns.scatterplot(
        x='log_frequency', 
        y='log_need_odds', 
        data=freq_stats_log, 
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Log Transformation: Need Odds vs Frequency')
    axes[1, 1].set_xlabel('Log Frequency')
    axes[1, 1].set_ylabel('Log Need Odds')
    axes[1, 1].grid(True)

    # --- Linear Regression for Log-Transformed Frequency Stats ---
    log_y_data = freq_stats_log[freq_stats_log['need_odds'] > 0].copy()
    log_y_data['log_need_odds'] = np.log(log_y_data['need_odds'])

    slope_fs, intercept_fs, r_value_fs, _, _ = linregress(
        log_y_data['log_frequency'], 
        log_y_data['log_need_odds']
    )

    sorted_log_freq_data = log_y_data.sort_values('log_frequency')
    sns.scatterplot(
        x='log_frequency', 
        y='log_need_odds',
        data=sorted_log_freq_data,
        ax=axes[1, 2]
    )

    x_values = sorted_log_freq_data['log_frequency']
    axes[1, 2].plot(
        x_values, 
        slope_fs * x_values + intercept_fs, 
        color='red', 
        linestyle='-', 
        label=f'y = {slope_fs:.2f}x + {intercept_fs:.2f}\nR² = {r_value_fs**2:.3f}'
    )

    axes[1, 2].legend()
    axes[1, 2].set_title('Log Need Odds vs Log Frequency (Regression Line)')
    axes[1, 2].set_xlabel('Log Frequency')
    axes[1, 2].set_ylabel('Log Need Odds')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()