# Import packages
from moderator import Moderator
from player import Player, RandomPlayer
from arena import Arena
import os

# Initialize results paths
if not os.path.exists(os.getcwd() + '/checkpoints'):
    os.mkdir(os.getcwd() + '/checkpoints')
if not os.path.exists(os.getcwd() + '/results'):
    os.mkdir(os.getcwd() + '/results')

# Initialize players
p00 = Player(name='Pietje Puk')
p01 = Player(name='Liesje Lot', bootstrapValueEpsilon=0.0001)
p02 = Player(name='Naima Nima', bootstrapValueEpsilon=0.01)
p03 = Player(name='Samir Smit', layers=[25])
p04 = Player(name='Biesje Bos', layers=[50,50,50])
p05 = Player(name='Fatima Flo', learningRate=0.0001)
p06 = Player(name='Omari Oost', learningRate=0.000001)
p07 = Player(name='Lida Leeuw', discountFactor=0.995)
p08 = Player(name='Kim Klasen', discountFactor=0.95)
p09 = Player(name='Diego Delo', layers=[250,250])

### TRAINING PROGRAM ###
# Training phase 1: everyone plays 100.000 random games against Randy Rado, learning 1.000x every 10.000 games, but not updating epsilon
for i, p in enumerate([p00, p01, p02, p03, p04, p05, p06, p07, p08, p09]):
    pr = RandomPlayer(name='Randy Rado')
    mdrtr = Moderator([p, pr])
    arn   = Arena(mdrtr, training_phase="part1")
    arn.play_and_learn()
    exec("del p"+str(i))
    del pr
    del mdrtr
    del arn
