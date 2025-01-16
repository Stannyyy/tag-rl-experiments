# Import packages
import random
import numpy as np
import pandas as pd
from moderator import Moderator
from arena import Arena
from config import retrieve_config

# Extract all variables from the config file
config = retrieve_config()

# Start a competition and m
def competition(game,players):

    # Set players reward store to 0
    for p in range(len(players)):
        players[p]._reward_store = []

    # Now loop over all combinations
    round_x_list = list(range(len(players)))
    round_y_list = list(range(len(players)))
    random.shuffle(round_x_list)
    random.shuffle(round_y_list)
    count = 1
    for round_x in round_x_list:
        for round_y in round_y_list:

            if round_x > round_y:

                # Announce game
                print('===')
                print('Game',count,'out of',int((len(players)*(len(players) - 1))/2))
                print('Players',players[round_x]._name,'and',players[round_y]._name,'are playing in the arena')

                # Set up moderator and arena
                modertr = Moderator(game,players=[players[round_x],players[round_y]])
                arena = Arena(modertr)

                # Compete in arena
                arena.play_and_learn()

                # Extract players
                players[round_x] = arena.modertr._players[0]
                players[round_y] = arena.modertr._players[1]

                count += 1

    # Rank and select top 2
    ranked_players = [round(np.mean(p._model._losses[-config['PRINT_EVERY']:]),5) for p in players]
    rewards = [np.sum(p._reward_store) for p in players]

    # Create pandas to order
    ranked_players = pd.DataFrame({'Players':range(len(players)),'Player namers':[p._name for p in players],'Losses':ranked_players,'Rewards':rewards}).sort_values('Rewards',ascending=False)
    winning_players = list(ranked_players.Players[0:4].values)

    # Print scoreboard of current generation
    print('The score of this generation was:')
    print(ranked_players)

    return players, winning_players, ranked_players