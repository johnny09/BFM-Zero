import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("train_log.txt")

n = len(df.columns)
cols = math.ceil(math.sqrt(n))
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].plot(df.index, df[col], linewidth=0.8)
    axes[i].set_title(col, fontsize=8)
    axes[i].tick_params(labelsize=6)

for i in range(n, len(axes)):
    axes[i].set_visible(False)

fig.tight_layout()
fig.savefig("results/bfmzero-isaac/train_log.png", dpi=150)
plt.show()
print(f"Saved to results/bfmzero-isaac/train_log.png  ({n} columns, {len(df)} rows)")
