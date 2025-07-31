import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from merge_with_emotnorm_dataset import word_stats_emot, freq_stats_emot
import pandas as pd
from patsy.contrasts import Treatment

df = word_stats_emot.copy()
df['valence_dm'] = df['valence'] - df['valence'].mean()  # demean valence
df['arousal_dm'] = df['arousal'] - df['arousal'].mean()  # demean arousal

valence_labels = ['Low Valence', 'Mid Valence', 'High Valence']
arousal_labels = ['Low Arousal', 'Mid Arousal', 'High Arousal']

df['valence_group'] = pd.qcut(df['valence'], q=3, labels=valence_labels)
df['arousal_group'] = pd.qcut(df['arousal'], q=3, labels=arousal_labels)

# Fit models
model1 = smf.ols('log_need ~ log_freq * valence_dm', data=df).fit() # interaction between log_freq and valence
model2 = smf.ols('log_need ~ log_freq * arousal_dm', data=df).fit() # interaction between log_freq and arousal
model3 = smf.ols('log_need ~ log_freq * valence_dm * arousal_dm', data=df).fit() # interaction between log_freq, valence, and arousal
model4 = smf.ols('log_need ~ log_freq + valence_dm + arousal_dm', data=df).fit() # main effects
model5 = smf.ols('log_need ~ log_freq * valence_dm + log_freq * arousal_dm', data=df).fit() # main effects with interaction terms
# model5 = smf.ols('log_need ~ log_freq * valence_dm * arousal_dm + I(valence_dm ** 2) * (log_freq + arousal_dm) + I(arousal_dm ** 2) * (log_freq + valence_dm)  ', data=df).fit() # higher-order interaction
model6 = smf.ols('log_need ~ log_freq * C(valence_group, Treatment(1)) * C(arousal_group, Treatment(1))', data=df).fit() # interaction between log_freq, valence, and arousal

summary = summary_col(
    [model4, model1, model2, model3, model5],
    model_names=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],
    stars=True,
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}"
        # 'N': lambda x: f"{int(x.nobs)}",
    },
    include_r2 = False
)

print(summary)
print(model6.params)

