import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load data
df = fetch_california_housing(as_frame=True).frame

# Histograms
df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.show()

# Boxplots
df.plot.box(figsize=(12, 8), subplots=True, layout=(3, 4), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# Outlier detection
outliers = {}
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outliers[col] = int(count)

print("Outlier counts per feature:")
print(outliers)
