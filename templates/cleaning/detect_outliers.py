DETECT_OUTLIERS = """
import numpy as np

z_scores = np.abs((df - df.mean()) / df.std())
outliers = (z_scores > 3).sum()
print("Outliers detected per column:")
print(outliers)
"""