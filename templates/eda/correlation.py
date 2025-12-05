CORRELATION = """
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False)
plt.title("Correlation matrix")
plt.show()
"""