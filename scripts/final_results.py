# Import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load data
experiment = 'experiment-20250820-1929'
df_paths = glob.glob(r'C:\Users\Stann\PycharmProjects\tag-rl-experiment/'+experiment+r'\competition\*\*\results.csv')
frames = []
for path in df_paths:
    try:
        df_temp = pd.read_csv(path)
        df_temp['source_file'] = os.path.basename(path)  # optional: keep track of origin
        frames.append(df_temp)
    except Exception as e:
        print(f"Skipping {path}: {e}")

# Concatenate into one DataFrame
df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
df = df.drop_duplicates()

# Add overall
df['player1_player_median'] = df['player1_tagger_median'] + df['player1_runner_median']
df['player2_player_median'] = df['player2_tagger_median'] + df['player2_runner_median']

# Fill nan values of same player competition
df.loc[df.player1==df.player2, 'player2_runner_median'] = df.loc[df.player1==df.player2, 'player1_runner_median']
df.loc[df.player1==df.player2, 'player2_tagger_median'] = df.loc[df.player1==df.player2, 'player1_tagger_median']
df.loc[df.player1==df.player2, 'player2_player_median'] = df.loc[df.player1==df.player2, 'player1_player_median']

# Add mirrored version
df_mirror = df.rename(columns={'player1': 'player2',
                               'player2': 'player1',
                               'player2_runner_median': 'player1_runner_median',
                               'player1_runner_median': 'player2_runner_median',
                               'player2_tagger_median': 'player1_tagger_median',
                               'player1_tagger_median': 'player2_tagger_median',
                               'player2_player_median': 'player1_player_median',
                               'player1_player_median': 'player2_player_median',})
pivot_cols = ['player1', 'player2',
              'player1_runner_median', 'player2_runner_median',
              'player1_tagger_median', 'player2_tagger_median',
              'player1_player_median', 'player2_player_median']
df_pivot = pd.concat((df[pivot_cols], df_mirror[pivot_cols]))
df_pivot = df_pivot.drop_duplicates()

# Create pivot tables
runner_pivot = df_pivot.pivot(index='player1', columns='player2', values='player1_runner_median')
tagger_pivot = df_pivot.pivot(index='player1', columns='player2', values='player1_tagger_median')
player_pivot = df_pivot.pivot(index='player1', columns='player2', values='player1_player_median')

# Sort
df_ = df.groupby(['player1'])['player1_player_median', 'player1_tagger_median', 'player1_runner_median'].median()
print(df_.sort_values('player1_player_median'))
df_[['player1_player_median', 'player1_tagger_median', 'player1_runner_median']] = (df_[['player1_player_median', 'player1_tagger_median', 'player1_runner_median']] - df_[['player1_player_median', 'player1_tagger_median', 'player1_runner_median']].median()) / df_[['player1_player_median', 'player1_tagger_median', 'player1_runner_median']].std()
print(df_.sort_values('player1_player_median'))

# Sort players for consistent axes
players_sorted = list(df_.sort_values('player1_player_median').index)
runner_pivot = runner_pivot.reindex(index=players_sorted, columns=players_sorted)
tagger_pivot = tagger_pivot.reindex(index=players_sorted, columns=players_sorted)
player_pivot = player_pivot.reindex(index=players_sorted, columns=players_sorted)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(30, 10), constrained_layout=True)

# Runner median heatmap
im1 = axes[0].matshow(runner_pivot, cmap='Greys')
axes[0].set_title("Runner median", fontsize=10)
axes[0].set_xticks(range(len(players_sorted)))
axes[0].set_yticks(range(len(players_sorted)))
axes[0].set_xticklabels(players_sorted, rotation=90, fontsize=6)
axes[0].set_yticklabels(players_sorted, fontsize=6)
axes[0].tick_params(axis='both', which='major', labelsize=6)
cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=6)

# Tagger median heatmap
im2 = axes[1].matshow(tagger_pivot, cmap='Greys')
axes[1].set_title("Tagger median", fontsize=10)
axes[1].set_xticks(range(len(players_sorted)))
axes[1].set_yticks(range(len(players_sorted)))
axes[1].set_xticklabels(players_sorted, rotation=90, fontsize=6)
axes[1].set_yticklabels(players_sorted, fontsize=6)
axes[1].tick_params(axis='both', which='major', labelsize=6)
cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=6)

# Overall median heatmap (player_median)
im3 = axes[2].matshow(player_pivot, cmap='Greys')
axes[2].set_title("Player median", fontsize=10)
axes[2].set_xticks(range(len(players_sorted)))
axes[2].set_yticks(range(len(players_sorted)))
axes[2].set_xticklabels(players_sorted, rotation=90, fontsize=6)
axes[2].set_yticklabels(players_sorted, fontsize=6)
axes[2].tick_params(axis='both', which='major', labelsize=6)
cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
cbar3.ax.tick_params(labelsize=6)

plt.savefig(os.path.join(os.getcwd(), experiment, "competition", "heatmaps.png"),
            dpi=300, bbox_inches='tight')
plt.close()

# Histograms
# Choose the row you want to visualize (by index or via a condition)
row_idx = 1
row = df.iloc[row_idx]

cols = [
    "player1_tagger_all",
    "player1_runner_all",
    "player2_tagger_all",
    "player2_runner_all",
]

plt.figure(figsize=(9, 6))

labels = [
    "Player 1 - Tagger",
    "Player 1 - Runner",
    "Player 2 - Tagger",
    "Player 2 - Runner",
]

colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

plt.hist(eval(row['player1_tagger_all']), bins=30, alpha=0.35, color=colors[0], edgecolor="white", label=labels[0])
plt.hist(eval(row['player1_runner_all']), bins=30, alpha=0.35, color=colors[1], edgecolor="white", label=labels[1])
plt.hist(eval(row['player2_tagger_all']), bins=30, alpha=0.35, color=colors[2], edgecolor="white", label=labels[2])
plt.hist(eval(row['player2_runner_all']), bins=30, alpha=0.35, color=colors[3], edgecolor="white", label=labels[3])

plt.title(f"Overlaid Histograms for Row {row_idx}")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
