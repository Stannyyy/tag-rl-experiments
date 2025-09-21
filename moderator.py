# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:39 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
import datetime
from config import Config
from game import Game
from memory import Memory

# Moderate game
class Moderator(Config):

    def __init__(self, players, experiment="defaultname"):

        # Import config
        Config.__init__(self)

        # Moderator variables
        self._experiment = experiment
        self._players = players
        self._randomPlayers = [p.isRandom for p in players]
        self._order_turns = random.sample(range(self.numPlayers), k=self.numPlayers)
        self._turn = self._order_turns[0]
        self._turns = []
        self._turn_count = 0
        self._turn_counts = []
        self._game = Game(experiment = experiment)
        self._games = []

    def get_players(self):
        return self._players
    players = property(get_players)

    def shuffle_players(self):
        random.shuffle(self._players)

    def alternate_players(self):
        self._players.reverse()

    def next_turn(self):
        idx = self._order_turns.index(self._turn)
        idx += 1
        if idx == len(self._order_turns):
            idx = 0
        self._turn = self._order_turns[idx]

    def play_one(self, save_game = False, learn = True):
        
        # Initialize game
        self._turn_count = 0
        self._game.init_random_game()

        # Reset players
        [p.new_game() for p in self._players]

        # Render
        if save_game:
            self._game.render()
            self._game.save(prefix=str(self._turn_count))

        # Start game
        game_over = False
        while game_over == False:

            # Determine next state
            self.next_turn()
            player = self._players[self._turn]
            player.options = self._game.what_options(self._turn)

            # Set new player state
            player.state = (self._game._x_list,
                            self._game._y_list,
                            self._turn,
                            self._game._taggers[self._turn])

            if learn:
                # Append next state and its options to sample (idx 3 and 4 of sample)
                player._memory.update_sample(player.options, player.state)
                player._memory.add_corrected_sample(self._game._tag_happened)

            # Make a move! Unless a tag has happened or the game has reached maximum turns
            if self._game._ended > 0:
                choice = 8
            else:
                choice = player.choose_action(player.options, save_game)
            reward = self._game.move(self._turn, choice)
            if self._turn_count > (self.maxSteps + self.numPlayers):
                raise Exception("Game is stuck in non-ending state.")
            self._turn_count += 1

            # Render
            if save_game:
                self._game.render()

            # Set new sample: state, choice, reward (idx 0, 1, 2 of sample)
            player._reward = reward
            if self._game._taggers[self._turn]:
                player._tot_reward_tagger += reward
            else:
                player._tot_reward_runner += reward
            if learn:
                player._memory.sample = (choice, reward, player.state)

            # Write if save video
            if save_game:
                self._game.save(prefix=str(self._turn_count+1))

            # Determine if game ended and determine next state
            if self._turn_count >= self.maxSteps:
                self._game._ended += 1
                player.options = self._game.what_options(self._turn)

            game_over = self._game._ended >= self.numPlayers

        # Create video
        if save_game:
            game_name = ' is playing against '.join([p._name for p in self._players]) + ' on ' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            self._game.record(game_name)

        # Update players
        unique_players = list(set(self._players))
        for player in unique_players:

            if learn:
                # Append next state and its options to sample (idx 3 and 4 of sample)
                if self._game._tag_happened:
                    player._memory.update_sample(None, player.state)
                else:
                    player._memory.update_sample(player.options, player.state)

                # Add last sample
                player._memory.add_corrected_sample(self._game._tag_happened)

                # Learn!
                player.learn_by_replay(int(self.batchSize * (len(self._players)/len(unique_players))))

            # Add rewards to reward store
            player.update_reward_store()

            # Add rewards to tensorboard
            if not save_game:
                player.add_rewards_to_tensorboard(self._turn_count)

            # Reset player
            player._tot_reward_tagger = 0
            player._tot_reward_runner = 0

        # Shuffle players for robustness
        if learn:
            self.shuffle_players()
        else:
            self.alternate_players()

    def play_many(self, learn=True):

        # Initialize games
        self._games = [Game(experiment = self._experiment) for ep in range(self.numEpisodesPerRound)]
        [self._games[ep].init_random_game() for ep in range(self.numEpisodesPerRound)]
        self._turn_counts = [0 for ep in range(self.numEpisodesPerRound)]
        game_overs = [False for ep in range(self.numEpisodesPerRound)]

        # Shuffle players for robustness
        self.shuffle_players()

        # Reset players
        [p.new_game() for p in self._players]
        unique_players = list(set(self._players))
        for p in unique_players:
            p.memory_many = [Memory(self.maxMemory) for ep in range(self.numEpisodesPerRound)]

        for i in range(self.maxSteps+self.numPlayers):

            # Print progress
            print(f"\rPlaying many games, step {i} out of {self.maxSteps}", end='')

            # Determine next state
            player = self._players[self._turn]
            player.options = np.array([[False, False, False, False, False, False, False, False, True] if game_overs[ep] else self._games[ep].what_options(self._turn) for ep in range(self.numEpisodesPerRound)])

            # Set new player state
            player.state_many = [(self._games[ep]._x_list,
                                  self._games[ep]._y_list,
                                  self._turn,
                                  self._games[ep]._taggers[self._turn])
                                  for ep in range(self.numEpisodesPerRound)]

            # Append next state and its options to sample (idx 3 and 4 of sample)
            [player.memory_many[ep].update_sample(player.options[ep], player.state_many[ep])
             for ep in range(self.numEpisodesPerRound)]
            [player.memory_many[ep].add_corrected_sample(self._games[ep]._tag_happened)
             for ep in range(self.numEpisodesPerRound)]

            # Select unfinished games
            slct = [ep for ep in range(self.numEpisodesPerRound) if not game_overs[ep]]

            # Make a move! Unless a tag has happened or the game has reached maximum turns
            choices = player.choose_many_actions(player.options, slct)
            choices = [8 if self._games[ep]._ended > 0 else choices[ep] for ep in range(self.numEpisodesPerRound)]

            # Set new sample
            rewards = [self._games[ep].move(self._turn, choices[ep]) for ep in range(self.numEpisodesPerRound)]
            player._rewards = rewards
            for ep in range(self.numEpisodesPerRound):
                if self._games[ep]._taggers[self._turn]:
                    player._tot_reward_tagger += rewards[ep]
                else:
                    player._tot_reward_runner += rewards[ep]
                player.memory_many[ep].sample = (choices[ep], rewards[ep], player.state_many[ep])

            # Determine if game ended and determine next state
            if i >= self.maxSteps:
                for ep in range(self.numEpisodesPerRound):
                    self._games[ep]._ended += 1
            game_overs = [self._games[ep]._ended >= self.numPlayers
                          for ep in range(self.numEpisodesPerRound)]

            # Next turn
            self.next_turn()
            self._turn_counts = [self._turn_counts[ep] if self._games[ep]._ended else self._turn_counts[ep] + 1 for ep in range(self.numEpisodesPerRound)]

        print("\nPlaying over, now preparing to learn")
        p._cnt += self.numEpisodesPerRound
        for p in unique_players:

            # Load all games into one memory
            for ep in range(self.numEpisodesPerRound):

                # Load samples to memory
                ep_samples = p.memory_many[ep].samples
                unique_ep_samples = []
                count_end_states = 0
                for ep_sample in ep_samples:
                    if ep_sample[3] is None:
                        count_end_states += 1
                    if (ep_sample not in unique_ep_samples) & (count_end_states <= self.numPlayers):
                        unique_ep_samples.append(ep_sample)
                        p._memory._sample = ep_sample
                        p._memory.add_sample()

            # Add rewards
            p.update_reward_store()
            p.add_rewards_to_tensorboard(np.mean(self._turn_counts))

            # Learn!
            if learn:
                prev_loss = 1000; min_learn_cycles = 10; learn_cycle = 0
                while True:
                    p.learn_by_replay(len(p._memory._samples), epochs=1, verbose=True)
                    p.step +=1
                    loss = [c.get('value') for c in p._summary_writer_collection if c.get('name') == 'params/losses'][-1]
                    learn_cycle += 1
                    print(f"Prev {prev_loss} now {loss}, with learn cycle {learn_cycle}")
                    if prev_loss < loss:
                        if learn_cycle > min_learn_cycles:
                            break
                    prev_loss = loss


    def write_video_text(self):
        text = 'Is tagger info: ' + str([i for i, x in enumerate(self._game._taggers) if x][0]) + '\n' + \
                'Reward player 0: ' + str(self._players[0]._reward) + '\n' + \
                'Reward player 1: ' + str(self._players[1]._reward) + '\n' + \
                'Total reward player 0 as tagger: ' + str(self._players[0]._tot_reward_tagger) + '\n' + \
               'Total reward player 0 as runner: ' + str(self._players[0]._tot_reward_runner) + '\n' + \
               'Total reward player 1 as tagger: ' + str(self._players[1]._tot_reward_tagger) + '\n' + \
               'Total reward player 1 as runner: ' + str(self._players[1]._tot_reward_runner) + '\n' + \
               'Turns: ' + str(self._turn_count)
        return text

    def describe_choice(self, choice):
        print({False:'o',True:'x'}.get(self._game._taggers[self._turn]))
        print({0:'up', 1:'down', 2:'left', 3:'right',
               4:'up left', 5:'up right', 6:'down left', 7:'down right'}.get(choice))

    def describe_sample(self, sample):
        print("Sample: ", sample)
        print("Role: ", {0: 'runner', 1: 'tagger'}.get(sample[0][5]))
        print("Turn: ", {0: 'left', 1: 'right'}.get(sample[0][4]))
        print("Left x,y: ", sample[0][0], sample[0][2])
        print("Right x,y: ", sample[0][1], sample[0][3])
        print("Choice: ", {0:'up', 1:'down', 2:'left', 3:'right',
               4:'up left', 5:'up right', 6:'down left', 7:'down right'}.get(sample[1]))
        print("Reward: ", sample[2])
