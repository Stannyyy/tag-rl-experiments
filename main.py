# Import packages
from moderator import Moderator
from player import Player, RandomPlayer
from arena import Arena

# Initialize players
p00 = RandomPlayer(name='Randy Rado')
p01 = Player(name='Pietje Puk')
p02 = Player(name='Liesje Lot', bootstrapValueEpsilon=0.0001)
p03 = Player(name='Naima Nima', bootstrapValueEpsilon=0.01)
p04 = Player(name='Samir Smit', layers=[25])
p05 = Player(name='Biesje Bos', layers=[50,50,50])
p06 = Player(name='Fatima Flo', learningRate=0.001)
p07 = Player(name='Omari Oost', learningRate=0.00001)
p08 = Player(name='Lida Leeuw', discountFactor=0.995)
p09 = Player(name='Kim Klasen', discountFactor=0.95)
p10 = Player(name='Diego Delo', layers=[250,250])

### TRAINING ###
# Training part 1: everyone plays 100.000 random games against Randy Rado, learning 1.000x every 10.000 games, but not updating epsilon
for p in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    mdrtr = Moderator([p, p00])
    arn   = Arena(mdrtr, update_epsilon_after=100000, nr_learning_episodes=1000, training_phase="part1")
    arn.play_and_learn()
