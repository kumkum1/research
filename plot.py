import matplotlib.pyplot as plt
import seaborn as sns
from metrics import word_stats

sns.scatterplot(data=word_stats[1:], x='frequency', y='recall_probability')
plt.title('Avg Frequency vs Recall Probability')
plt.xlabel('Average Frequency')
plt.ylabel('Recall Probability')
plt.grid(True)
plt.show()
