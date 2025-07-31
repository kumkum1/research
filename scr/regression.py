import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import pandas as pd
from patsy.contrasts import Treatment

# Load the datasets
df = pd.read_csv('data/merged/word_stats_emot.csv')

df['valence_dm'] = df['valence'] - df['valence'].mean()  
df['arousal_dm'] = df['arousal'] - df['arousal'].mean()  

valence_labels = ['Low Valence', 'Mid Valence', 'High Valence']
arousal_labels = ['Low Arousal', 'Mid Arousal', 'High Arousal']
df['valence_group'] = pd.qcut(df['valence'], q=3, labels=valence_labels)
df['arousal_group'] = pd.qcut(df['arousal'], q=3, labels=arousal_labels)

# Fit Models
model1 = smf.ols('log_need ~ log_freq * valence_dm', data=df).fit() # interaction between log_freq and valence
model2 = smf.ols('log_need ~ log_freq * arousal_dm', data=df).fit() # interaction between log_freq and arousal
model3 = smf.ols('log_need ~ log_freq * valence_dm * arousal_dm', data=df).fit() # interaction between log_freq, valence, and arousal
model4 = smf.ols('log_need ~ log_freq + valence_dm + arousal_dm', data=df).fit() # main effects
model5 = smf.ols('log_need ~ log_freq * valence_dm + log_freq * arousal_dm', data=df).fit() # main effects with interaction terms
# model6 = smf.ols('log_need ~ log_freq * valence_dm * arousal_dm + I(valence_dm ** 2) * (log_freq + arousal_dm) + I(arousal_dm ** 2) * (log_freq + valence_dm)  ', data=df).fit() 
model7 = smf.ols('log_need ~ log_freq * C(valence_group, Treatment(1)) * C(arousal_group, Treatment(1))', data=df).fit() 

summary = summary_col(
    [model4, model1, model2, model3, model5],
    model_names=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],
    stars=True,
    info_dict={
        'R-squared': lambda x: f"{x.rsquared:.3f}",
        'N': lambda x: f"{int(x.nobs)}"
    },
    include_r2 = False,
)

# print(summary)
# print(model7.summary())

with open("output/regression_tables/multiple_regression_summary.txt", "w") as f: f.write(summary.as_text())
with open("output/regression_tables/3split_interaction.txt", "w") as f: f.write(str(model7.summary()))
