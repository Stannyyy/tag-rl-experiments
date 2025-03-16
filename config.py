# SET VARIABLES
# Reinforcement learning model variables
class Config:
    def __init__(self,
                 maxEpsilon = 1, minEpsilon = 0.01, batchSize = 100, maxMemory = 5000,
                 gridSize = 10,
                 numPlayers = 2, numTaggers = 1,
                 renderSpeed = 0.3, createVideo = True,
                 numEpisodes = 10000, numEpisodesBeforePrint = 100, minLoss = 1, numGamesShown = 100
                ):

        # Model config
        self.maxEpsilon = maxEpsilon
        self.minEpsilon = minEpsilon
        self.batchSize = batchSize
        self.maxMemory = maxMemory
        
        # Game config
        self.gridSize = gridSize
        self.numPlayers = numPlayers
        self.numTaggers = numTaggers
        self.numStates = 2 * numPlayers + 2
        self.numActions = 9
        
        # Match config
        self.numEpisodes = numEpisodes
        self.numEpisodesBeforePrint = numEpisodesBeforePrint
        self.minLoss = minLoss  # Stop when loss < minLoss
        
        # Render config
        self.renderSpeed = renderSpeed  # Players move every ~ seconds in rendering window
        self.numGamesShown = numGamesShown
        self.createVideo = createVideo
