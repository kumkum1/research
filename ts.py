import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from metrics import word_stats, freq_stats, recall_df         
from combine_data import cleaned_df
from scipy.stats import linregress
import statsmodels.formula.api as smf


# print(cleaned_df)
#freq_stats -> [1893 rows x 6 columns]

emot_df = pd.read_csv('emot_28724.csv')
emot_df.rename(columns=lambda x: x.lower().strip(), inplace=True)
emot_df = emot_df[['word', 'valence', 'arousal']]

merged_df = pd.merge(word_stats, emot_df, on='word', how='left')
merged_df.to_csv('combined_with_emotion.csv', index=False)

# data with only complete rows for emotion analysis
word_stats_emot = merged_df.dropna(subset=['valence', 'arousal'])
print(word_stats_emot)

# --------------------------------------------- ##

# recall_df['word'] = recall_df['word'].replace('mln', 'million')
# recall_df['word'] = recall_df['word'].replace('pct', 'percent')

recalled_with_emotion = (
    recall_df.merge(emot_df, on='word', how='left')
)

# print(recall_df)
# print(recalled_with_emotion)

freq_stats_emot = recalled_with_emotion.groupby('frequency').agg(
    total=('is_recalled', 'count'),
    recalls=('is_recalled', 'sum'),
    mean_valence=('valence', 'mean'),
    mean_arousal=('arousal', 'mean')
).reset_index()
freq_stats_emot['recall_probability'] = freq_stats_emot['recalls'] / freq_stats_emot['total']
freq_stats_emot['need_odds'] = freq_stats_emot['recall_probability'].transform(lambda x: x/(1-x))
freq_stats_emot.dropna(subset=['mean_valence', 'mean_arousal'], inplace=True)

print(freq_stats_emot)
# print(freq_stats)

# --- Normalization Helper ---
def norm(series):
    return (series - series.min()) / (series.max() - series.min())

# --- Word-level Plot Data ---
word_stats_emot = word_stats_emot.copy()
word_stats_emot = word_stats_emot[(word_stats_emot['avg_frequency'] > 0) & (word_stats_emot['need_odds'] > 0)]

word_stats_emot['log_freq'] = np.log(word_stats_emot['avg_frequency'])
word_stats_emot['log_need'] = np.log(word_stats_emot['need_odds'])
word_stats_emot['arousal_n'] = norm(word_stats_emot['arousal'])

# --- Frequency-level Plot Data ---
freq_stats_emot = freq_stats_emot.copy()
freq_stats_emot = freq_stats_emot[(freq_stats_emot['frequency'] > 0) & (freq_stats_emot['need_odds'] > 0)]

freq_stats_emot['log_freq'] = np.log(freq_stats_emot['frequency'])
freq_stats_emot['log_need'] = np.log(freq_stats_emot['need_odds'])
freq_stats_emot['arousal_n'] = norm(freq_stats_emot['mean_arousal'])

print(freq_stats_emot)
# # --------------------------------------------------  PLOTS  ------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# vmin = min(word_stats_emot['valence'].min(), freq_stats_emot['mean_valence'].min())
# vmax = max(word_stats_emot['valence'].max(), freq_stats_emot['mean_valence'].max())

vmin = 1
vmax = 9

# (0,0) WORD LEVEL
sc1 = axes[0].scatter(
    word_stats_emot['log_freq'], word_stats_emot['log_need'],
    s = 20 + 380 * word_stats_emot['arousal_n'],
    c = word_stats_emot['valence'],
    cmap = 'viridis',
    alpha = 0.25 + 0.75 * word_stats_emot['arousal_n'],
    edgecolor='k', linewidths=0,
    vmin=vmin, vmax=vmax
)
axes[0].set_title('Group by Word: Arousal→Size/Opacity, Valence→Colour')
axes[0].set_ylabel('Log Need Odds')
fig.colorbar(sc1, ax=axes[0], label='Valence')

#regression 
slope, intercept, r, _, _ = linregress(word_stats_emot['log_freq'], word_stats_emot['log_need'])
x_vals = np.sort(word_stats_emot['log_freq'])
axes[0].plot(x_vals, slope * x_vals + intercept, color='red', linestyle='-', linewidth=1.5)
axes[0].legend([f'y={slope:.2f}x+{intercept:.2f}\nR²={r**2:.2f}'], loc='lower right', fontsize='x-small')


# (1,0) FREQ LEVEL
sc2 = axes[1].scatter(
    freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
    s = 20 + 380 * freq_stats_emot['arousal_n'],
    c = freq_stats_emot['mean_valence'],
    cmap = 'viridis',
    alpha = 0.25 + 0.75 * freq_stats_emot['arousal_n'],
    edgecolor='k', linewidths=0,
    vmin=vmin, vmax=vmax
)
axes[1].set_title('Group by Freq: Arousal→Size/Opacity, Valence→Colour')
axes[1].set_xlabel('Log Frequency')
axes[1].set_ylabel('Log Need Odds')
fig.colorbar(sc2, ax=axes[1], label='Mean Valence')

#regression
slope, intercept, r, _, _ = linregress(freq_stats_emot['log_freq'], freq_stats_emot['log_need'])
x_vals = np.sort(freq_stats_emot['log_freq'])
axes[1].plot(x_vals, slope * x_vals + intercept, color='red', linestyle='-', linewidth=1.5)
axes[1].legend([f'y={slope:.2f}x+{intercept:.2f}\nR²={r**2:.2f}'], loc='lower right', fontsize='x-small')

# Add gridlines
for ax in axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# plt.tight_layout()
# plt.show()


# -- Valence and Arousal separated -----------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

# Word-level valence
sc0 = axes[0, 0].scatter(word_stats_emot['log_freq'], word_stats_emot['log_need'],
                         c=word_stats_emot['valence'], cmap='coolwarm', alpha=0.7,
                         vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Word: Valence ")
fig.colorbar(sc0, ax=axes[0, 0], label="Valence")

# Word-level arousal
sc1 = axes[1, 0].scatter(word_stats_emot['log_freq'], word_stats_emot['log_need'],
                         c=word_stats_emot['arousal'], cmap='coolwarm', alpha=0.7, vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Word: Arousal ")
fig.colorbar(sc1, ax=axes[1, 0], label="Arousal")

# Frequency-bucket valence
sc2 = axes[0, 1].scatter(freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
                         c=freq_stats_emot['mean_valence'], cmap='coolwarm', alpha=0.8,
                         vmin=vmin, vmax=vmax)
axes[0, 1].set_title("Freq: Valence ")
fig.colorbar(sc2, ax=axes[0, 1], label="Mean Valence")

# Frequency-bucket arousal
sc3 = axes[1, 1].scatter(freq_stats_emot['log_freq'], freq_stats_emot['log_need'],
                         c=freq_stats_emot['mean_arousal'], cmap='coolwarm', alpha=0.8, vmin=vmin, vmax=vmax)
axes[1, 1].set_title("Freq: Arousal ")
fig.colorbar(sc3, ax=axes[1, 1], label="Mean Arousal")

for ax in axes[1, :]:
    ax.set_xlabel("log Frequency")
for ax in axes[:, 0]:
    ax.set_ylabel("log Need Odds")

for row in axes:
    for ax in row:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


# -- Split valence into 3 groups ---------------------------------------------------------------

# Define valence and arousal groups
valence_labels = ['Low Valence', 'Mid Valence', 'High Valence']
arousal_labels = ['Low Arousal', 'Mid Arousal', 'High Arousal']

word_stats_emot['valence_group'] = pd.qcut(word_stats_emot['valence'], q=3, labels=valence_labels)
word_stats_emot['arousal_group'] = pd.qcut(word_stats_emot['arousal'], q=3, labels=arousal_labels)

# Create 9 subplots
fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True, sharey=True)

for i, valence_label in enumerate(valence_labels):
    for j, arousal_label in enumerate(arousal_labels):
        ax = axes[i, j]
        subset = word_stats_emot[(word_stats_emot['valence_group'] == valence_label) & (word_stats_emot['arousal_group'] == arousal_label)]
        
        sc = ax.scatter(
            subset['log_freq'], subset['log_need'],
            s=20 + 380 * subset['arousal_n'],
            c=subset['valence'], cmap='viridis', alpha=0.7,
            edgecolor='k', linewidths=0, vmin=1, vmax=9
        )
        ax.set_title(f'{valence_label} - {arousal_label}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Add regression line only if enough data
        if len(subset) >= 2 and subset['log_freq'].nunique() > 1:

            # 1. Multiple regression with interaction terms (OLS)
            model = smf.ols(
                formula='log_need ~ log_freq * valence + log_freq * arousal',
                data=subset
            ).fit()

            freq_range = np.linspace(subset['log_freq'].min(), subset['log_freq'].max(), 100)
            val_mean = subset['valence'].mean()
            aro_mean = subset['arousal'].mean()

            pred_df = pd.DataFrame({
                'log_freq': freq_range,
                'valence': val_mean,
                'arousal': aro_mean
            })
            pred_df['log_need_pred'] = model.predict(pred_df)

            [line_interaction] = ax.plot(
                freq_range, pred_df['log_need_pred'],
                color='blue', linestyle='--', linewidth=1.5,
                label=f'MultReg R²={model.rsquared:.2f}'
            )

            # 2. Simple linear regression
            slope, intercept, r_value, _, _ = linregress(subset['log_freq'], subset['log_need'])

            x_vals = np.sort(subset['log_freq'])
            y_vals = slope * x_vals + intercept

            [line_simple] = ax.plot(
                x_vals, y_vals,
                color='red', linestyle='-', linewidth=1.5,
                label=f'Simple y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.2f}'
            )

            # Combine both lines in legend
            ax.legend(handles=[line_simple, line_interaction], loc='lower right', fontsize='x-small')

fig.colorbar(sc, ax=axes, label='Valence', orientation='vertical', fraction=0.02, pad=0.04)

for ax in axes[-1, :]:
    ax.set_xlabel('Log Frequency')
for ax in axes[:, 0]:
    ax.set_ylabel('Log Need Odds')

############ --------------------------------------------------------------------------------- ############
############ --------------------------------------------------------------------------------- ############
############ --------------------------------------------------------------------------------- ############


# Create a single plot with different colors for valence and sizes for arousal
fig, ax = plt.subplots(figsize=(12, 8))

valence_colors = {'Low Valence': 'blue', 'Mid Valence': 'green', 'High Valence': 'red'}
arousal_sizes = {'Low Arousal': 50, 'Mid Arousal': 200, 'High Arousal': 500}

for valence_label in valence_labels:
    for arousal_label in arousal_labels:
        subset = word_stats_emot[(word_stats_emot['valence_group'] == valence_label) & (word_stats_emot['arousal_group'] == arousal_label)]
        
        ax.scatter(
            subset['log_freq'], subset['log_need'],
            s=arousal_sizes[arousal_label],
            c=valence_colors[valence_label],
            alpha=0.7, edgecolor='k', linewidths=0.5,
            label=f'{valence_label} - {arousal_label}'
        )

valid_data = word_stats_emot[(word_stats_emot['need_odds'] > 0)].copy()
if len(valid_data) >= 2 and valid_data['log_freq'].nunique() > 1:
    slope, intercept, r_value, _, _ = linregress(valid_data['log_freq'], valid_data['log_need'])
    x_vals = np.linspace(valid_data['log_freq'].min(), valid_data['log_freq'].max(), 200)
    y_vals = slope * x_vals + intercept
    
    ax.plot(x_vals, y_vals, color='black', linestyle='-', linewidth=2,
            label=f'Regression: y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.2f}')

ax.set_xlabel('Log Frequency')
ax.set_ylabel('Log Need Odds')
ax.set_title('Combined Plot: Valence (Color) and Arousal (Size)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Groups')

plt.tight_layout()
plt.show()
