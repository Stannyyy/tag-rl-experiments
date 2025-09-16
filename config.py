# SET VARIABLES
# Reinforcement learning model variables
class Config:
    def __init__(self,
                 maxEpsilon = 1, minEpsilon = 0.001, batchSize = 100, maxMemory = 5000,
                 gridSize = 10,
                 numPlayers = 2, numTaggers = 1,
                 renderSpeed = 0.3, createVideo = True,
                 numEpisodes = 1000000, numEpisodesPerRound = 10000, minLoss = 1, numGamesShown = 100
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
        self.stepPoints = 0.1 * gridSize
        self.tagPoints = 5.1 * gridSize
        self.maxSteps = 50
        
        # Match config
        self.numEpisodes = numEpisodes
        self.numEpisodesPerRound = numEpisodesPerRound
        self.minLoss = minLoss  # Stop when loss < minLoss
        
        # Render config
        self.renderSpeed = renderSpeed  # Players move every ~ seconds in rendering window
        self.numGamesShown = numGamesShown
        self.createVideo = createVideo

# config.py
class Config:
    def __init__(self, **kwargs):
        # Default values
        defaults = dict(
            # Training regime
            maxEpsilon=1,
            minEpsilon=0.01,
            numEpisodes=1000000,
            numEpisodesPerRound=10000,
            bootstrapValueEpsilon=0.001,
            discountFactor=0.99,
            preselectBatch=False,
            useProbabilities=False,

            # Model variables
            learningRate=0.0001,
            layers=[100, 100, 100],

            # Architecture options
            ## LSTM
            addLSTM=False,
            sequenceLengthLSTM=1,

            ## Curiosity bonus
            curiosity=False,
            curiosity_beta=0,

            # Memory variables
            batchSize=100, # Note these are only relevant in sequential mode
            maxMemory=5000, # Note these are only relevant in sequential mode

            # Game variables
            gridSize=10, # If this is changed, consider changing tagPoints
            numPlayers=2,
            numTaggers=1,
            maxSteps=50, # If this is changed, consider changing tagPoints
            stepPoints=1.0,
            tagPoints=51.0,
            # Note that code has to be changed to support more or less actions
            numActions=9,

            # Render variables
            renderSpeed=0.3,
            createVideo=True,

            # Test mode
            testMode=False,

            # If custom numStates/numActions (for testing)
            numStatesOverwrite=None,
            numActionsOverwrite=None
        )

        # Apply overrides from kwargs
        defaults.update(kwargs)

        # Set all attributes dynamically
        for key, value in defaults.items():
            setattr(self, key, value)

        # Derived attributes
        self.numStates = 2 * self.numPlayers + 2