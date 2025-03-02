# Import packages
from moderator import Moderator
from player import Player, RandomPlayer, StillPlayer
from arena import Arena
import os
import datetime
import tensorflow as tfbare

# Experiment
experiment = "/experiment-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize results paths
if not os.path.exists(os.getcwd() + experiment):
    os.mkdir(os.getcwd() + experiment)
if not os.path.exists(os.getcwd() + experiment + '/checkpoints'):
    os.mkdir(os.getcwd() + experiment + '/checkpoints')
if not os.path.exists(os.getcwd() + experiment + '/results'):
    os.mkdir(os.getcwd() + experiment + '/results')

# Initialize players
p00 = Player(experiment, name='Pietje Puk')
p01 = Player(experiment, name='Liesje Lot', bootstrapValueEpsilon=0.0001)
p02 = Player(experiment, name='Naima Nima', bootstrapValueEpsilon=0.01)
p03 = Player(experiment, name='Samir Smit', layers=[25])
p04 = Player(experiment, name='Biesje Bos', layers=[50,50,50])
p05 = Player(experiment, name='Fatima Flo', learningRate=0.0001)
p06 = Player(experiment, name='Omari Oost', learningRate=0.000001)
p07 = Player(experiment, name='Lida Leeuw', discountFactor=0.995)
p08 = Player(experiment, name='Kim Klasen', discountFactor=0.95)
p09 = Player(experiment, name='Diego Delo', layers=[250,250])

### TRAINING PROGRAM ###

# Training phase 1: everyone plays 100.000 random games against Stable Sef, learning 1.000x every 10.000 games, but not updating epsilon
for i, p in enumerate([p00,p01,p02,p03,p04,p05,p06,p07,p08,p09]):
    
    # Set up the tensorboard
    log_dir = experiment + "/logs/dql_" + p._name
    p._summary_writer = tfbare.summary.create_file_writer(log_dir)
    p._summary_loss_step = 0
    p._summary_params_step = 0
    p._summary_reward_step = 0
        
    pr = StillPlayer(name='Stable Sef')
    mdrtr = Moderator([p, pr], experiment = experiment)
    arn   = Arena(mdrtr, training_phase="part1")
    arn.play_and_learn()
    del pr
    del mdrtr
    del arn

    # Training phase 2: everyone plays 100.000 random games against Randy Rado, learning 1.000x every 10.000 games, but not updating epsilon
    pr = RandomPlayer(name='Randy Rado')
    mdrtr = Moderator([p, pr], experiment = experiment)
    arn   = Arena(mdrtr, training_phase="part2")
    arn.play_and_learn()
    del pr
    del mdrtr
    del arn
    
    # Close file writer
    p._summary_writer.close()
    del p