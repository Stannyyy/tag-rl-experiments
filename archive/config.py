# SET VARIABLES
# Reinforcement learning model variables
def retrieve_config():
    config = {'MAX_EPSILON'  : 1,
                'MIN_EPSILON'  : 0,
                'LAMBDA'       : 0.001,
                'ALPHA'        : 0.001,
                'GAMMA'        : 0.99,
                'BATCH_SIZE'   : 100,
                'MAX_MEMORY'   : 5000,

                # Game variables
                'GRID_SIZE'    : 10,

                # Player variables
                'NUM_PLAYERS'  : 2,
                'IS_RANDOM'    : [False,False], # List of len(NUM_PLAYERS) saying which are random
                'NUM_TAGGERS'  : 1,

                # Do you want to render the game (I advice to do this only after the model had time to form)
                'RENDER'       : False, # If True, every PRINT_EVERY episodes, an episode will be rendered
                'RENDER_SPEED' : 0.3, # Players move every ~ seconds in rendering window

                # Experiment variables
                'NUM_EPISODES' : 1000, # Max number of episodes to play, if you want to keep playing until MIN_LOSS is reached, make this number high
                'PRINT_EVERY'  : 100, # Shows a plot of rewards per player every ~ episodes
                'MIN_LOSS'     : 1, # Stop when loss < MIN_LOSS, if you want to keep playing until NUM_EPISODES, make this number high

                # Render episodes at the end of training
                'N_GAMES_SHOW' : 2000, # Render is always True here, if you don't want to render at the end, assign 0

                # Evolutionary config
                'CHANCE_OF_MUTATION' : 0.01
    }
    return config