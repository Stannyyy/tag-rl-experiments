# Import packages
import os
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load data
experiment = 'experiment-20250808-1204'
df = pd.read_csv(r'C:\Users\Stann\PycharmProjects\tag-rl-experiment/'+experiment+r'\final-results\results.csv')

# Create pivot tables
runner_pivot = df.pivot(index='player1', columns='player2', values='player1_runner_mean')
tagger_pivot = df.pivot(index='player1', columns='player2', values='player1_tagger_mean')

# Sort players for consistent axes
players_sorted = sorted(set(df['player1']) | set(df['player2']))
runner_pivot = runner_pivot.reindex(index=players_sorted, columns=players_sorted)
tagger_pivot = tagger_pivot.reindex(index=players_sorted, columns=players_sorted)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Runner heatmap
im1 = axes[0].matshow(runner_pivot, cmap='viridis')
axes[0].set_title("Runner Mean")
axes[0].set_xticks(range(len(players_sorted)))
axes[0].set_yticks(range(len(players_sorted)))
axes[0].set_xticklabels(players_sorted, rotation=90)  # Vertical column labels
axes[0].set_yticklabels(players_sorted)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Smaller legend

# Tagger heatmap
im2 = axes[1].matshow(tagger_pivot, cmap='plasma')
axes[1].set_title("Tagger Mean")
axes[1].set_xticks(range(len(players_sorted)))
axes[1].set_yticks(range(len(players_sorted)))
axes[1].set_xticklabels(players_sorted, rotation=90)  # Vertical column labels
axes[1].set_yticklabels(players_sorted)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Smaller legend

plt.tight_layout()
plt.savefig(os.getcwd()+'/'+experiment+ "/final-results/heatmaps.png", dpi=300, bbox_inches='tight')  # Save as PNG
plt.close()

# Order
df['player1_mean_all'] = df['player1_tagger_mean'] + df['player1_runner_mean']
df_ = df.groupby(['player1'])['player1_mean_all', 'player1_tagger_mean', 'player1_runner_mean'].mean()
print(df_.sort_values('player1_mean_all'))
df_[['player1_mean_all', 'player1_tagger_mean', 'player1_runner_mean']] = (df_[['player1_mean_all', 'player1_tagger_mean', 'player1_runner_mean']] - df_[['player1_mean_all', 'player1_tagger_mean', 'player1_runner_mean']].mean()) / df_[['player1_mean_all', 'player1_tagger_mean', 'player1_runner_mean']].std()
print(df_.sort_values('player1_mean_all'))