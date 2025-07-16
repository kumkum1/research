import pandas as pd
from data import text_df  


emot_df = pd.read_csv('emot_28724.csv')
emot_df.rename(columns=lambda x: x.upper().strip(), inplace=True)
emot_df = emot_df[['WORD', 'VALENCE', 'AROUSAL', 'DOMINANCE']]

merged_df = pd.merge(text_df, emot_df, on='WORD', how='left')
merged_df.to_csv('combined_with_emotion.csv', index=False)

# data with only complete rows for emotion analysis
cleaned_df = merged_df.dropna(subset=['VALENCE', 'AROUSAL', 'DOMINANCE'])
cleaned_df.to_csv("clean.csv", index=False)

# print('dollars' in emot_df['WORD'].values)