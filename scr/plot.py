import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress
import os

def plot_stats(word_stats, freq_stats):
    os.makedirs("output/plots", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    def log_transform(df, xcol, ycol):
        df = df[(df[xcol] > 0) & (df[ycol] > 0)].copy()
        df[f'log_{xcol}'] = np.log(df[xcol])
        df[f'log_{ycol}'] = np.log(df[ycol])
        return df

    def plot_scatter(ax, data, x, y, title, xlabel, ylabel):
        sns.scatterplot(x=x, y=y, data=data, ax=ax, alpha=0.5, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    def plot_regression(ax, data, x, y, title, xlabel, ylabel):
        slope, intercept, r_value, _, _ = linregress(data[x], data[y])
        sns.scatterplot(x=x, y=y, data=data, ax=ax, alpha=0.5, linewidth=0)
        ax.plot(
            data[x],
            slope * data[x] + intercept,
            color='red',
            linestyle='-',
            label=f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.3f}'
        )
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Word Plots 
    plot_scatter(axes[0, 0], word_stats, 'avg_frequency', 'need_odds',
                 'Need Odds vs Frequency (by Word)', 'Avg Frequency', 'Need Odds')

    word_stats_log = log_transform(word_stats, 'avg_frequency', 'need_odds')
    plot_scatter(axes[0, 1], word_stats_log, 'log_avg_frequency', 'log_need_odds',
                 'Log-Transformed: Need Odds vs Frequency', 'Log Avg Frequency', 'Log Need Odds')

    plot_regression(axes[0, 2], word_stats_log, 'log_avg_frequency', 'log_need_odds',
                    'Log Need Odds vs Log Frequency (Regression)', 'Log Avg Frequency', 'Log Need Odds')

    # Frequency Plots 
    plot_scatter(axes[1, 0], freq_stats, 'frequency', 'need_odds',
                 'Need Odds vs Frequency (by Frequency)', 'Frequency', 'Need Odds')

    freq_stats_log = log_transform(freq_stats, 'frequency', 'need_odds')
    plot_scatter(axes[1, 1], freq_stats_log, 'log_frequency', 'log_need_odds',
                 'Log-Transformed: Need Odds vs Frequency', 'Log Frequency', 'Log Need Odds')

    plot_regression(axes[1, 2], freq_stats_log, 'log_frequency', 'log_need_odds',
                    'Log Need Odds vs Log Frequency (Regression)', 'Log Frequency', 'Log Need Odds')

    plt.tight_layout()
    plt.savefig('output/plots/basic_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()