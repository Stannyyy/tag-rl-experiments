# Import packages
import os
import random
import numpy as np
import datetime
from game import Game
from memory import Memory

# Moderate game
class Moderator:

    def __init__(self, config, players, display_games_path="defaultname"):

        # Moderator variables
        self._config = config
        self._players = players
        self._order_of_turns = random.sample(range(config.number_of_players), k=config.number_of_players)
        self._turn = self._order_of_turns[0]
        self._turn_count = 0
        self._turn_counts = []
        self._display_games_path = display_games_path
        self._game = Game(config, display_games_path)
        self._games = []

    def shuffle_players(self):
        random.shuffle(self._players)

    def alternate_players(self):
        self._players.reverse()

    def next_turn(self):
        idx = self._order_of_turns.index(self._turn)
        idx += 1
        if idx == len(self._order_of_turns):
            idx = 0
        self._turn = self._order_of_turns[idx]

    def play_one(self, save_game = False, learn = True):
        
        # Initialize game
        self._turn_count = 0
        self._game.init_random_game()

        # Reset (unique) players
        [player.new_game() for player in list(set(self._players))]

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
            player.current_options = self._game.what_options(self._turn)

            # Set new player state
            player.state = (self._game.x_list,
                            self._game.y_list,
                            self._turn,
                            self._game.taggers[self._turn])

            if learn:
                # Append next state and its options to experience (idx 3 and 4 of experience)
                player.memory.update_experience(player.current_options, player.state)
                player.memory.add_corrected_experience(self._game.tag_happened)

            # Make a move! Unless a tag has happened or the game has reached maximum turns
            if self._game.ended > 0:
                choice = 8
            else:
                choice = player.choose_action(player.current_options, save_game)
            reward = self._game.move(self._turn, choice)
            if self._turn_count > (self._config.maximum_steps + self._config.number_of_players):
                raise Exception("Game is stuck in non-ending state.")
            self._turn_count += 1

            # Render
            if save_game:
                self._game.render()

            # Set new experience: state, choice, reward (idx 0, 1, 2 of experience)
            player.current_reward = reward
            if self._game.taggers[self._turn]:
                player._total_reward_tagger += reward
            else:
                player._total_reward_runner += reward
            if learn:
                player.memory.experience = (choice, reward, player.state)

            # Write if save video
            if save_game:
                self._game.save(prefix=str(self._turn_count))

            # Determine if game ended and determine next state
            if self._turn_count >= self._config.maximum_steps:
                self._game.ended += 1
                player.current_options = self._game.what_options(self._turn)

            game_over = self._game.ended >= self._config.number_of_players

        # Create video
        if save_game:
            game_path = os.path.join(
                *[player.name for player in self._players],
                datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            )
            self._game.record(game_path)

        # Update players
        unique_players = list(set(self._players))
        for player in unique_players:

            if learn:
                # Append next state and its options to experience (idx 3 and 4 of experience)
                if self._game.tag_happened:
                    player.memory.update_experience(None, player.state)
                else:
                    player.memory.update_experience(player.current_options, player.state)

                # Add last experience
                player.memory.add_corrected_experience(self._game.tag_happened)

                # Learn!
                player.learn_by_replay(int(self._config.batch_size * (len(self._players)/len(unique_players))))
                player.add_logs_to_tensorboard()

            # Add rewards to reward store
            player.update_reward_store()

            # Add rewards to tensorboard
            if not save_game:
                player.add_rewards_to_tensorboard(self._turn_count)

            # Reset player
            player._total_reward_tagger = 0
            player._total_reward_runner = 0

        # Shuffle players for robustness
        if learn:
            self.shuffle_players()
        else:
            self.alternate_players()

    def play_many(self, learn=True):

        # Initialize games
        self._games = [Game(self._config, self._display_games_path) for episode in range(self._config.number_of_episodes_per_round)]
        [self._games[episode].init_random_game() for episode in range(self._config.number_of_episodes_per_round)]
        self._turn_counts = [0 for episode in range(self._config.number_of_episodes_per_round)]
        game_overs = [False for episode in range(self._config.number_of_episodes_per_round)]

        # Shuffle players for robustness
        self.shuffle_players()

        # Reset players
        [player.new_game() for player in self._players]
        unique_players = list(set(self._players))
        for player in unique_players:
            player.memory_many = [Memory(self._config) for episode in range(self._config.number_of_episodes_per_round)]

        for i in range(self._config.maximum_steps+self._config.number_of_players):

            # Print progress
            print(f"\rPlaying many games, step {i} out of {self._config.maximum_steps}", end='')

            # Determine next state
            player = self._players[self._turn]
            player.current_options = np.array([[False, False, False, False, False, False, False, False, True]
                                               if game_overs[episode]
                                               else self._games[episode].what_options(self._turn)
                                               for episode in range(self._config.number_of_episodes_per_round)])

            # Set new player state
            player.state_many = [(self._games[episode].x_list,
                                  self._games[episode].y_list,
                                  self._turn,
                                  self._games[episode].taggers[self._turn])
                                  for episode in range(self._config.number_of_episodes_per_round)]

            # Append next state and its options to experience (idx 3 and 4 of experience)
            [player.memory_many[episode].update_experience(player.current_options[episode], player.state_many[episode])
             for episode in range(self._config.number_of_episodes_per_round)]
            [player.memory_many[episode].add_corrected_experience(self._games[episode].tag_happened)
             for episode in range(self._config.number_of_episodes_per_round)]

            # Select unfinished games
            selection = [episode for episode in range(self._config.number_of_episodes_per_round) if not game_overs[episode]]

            # Make a move! Unless a tag has happened or the game has reached maximum turns
            choices = player.choose_many_actions(player.current_options, selection)
            choices = [8 if self._games[episode].ended > 0 else choices[episode] for episode in range(self._config.number_of_episodes_per_round)]

            # Set new experience
            rewards = [self._games[episode].move(self._turn, choices[episode]) for episode in range(self._config.number_of_episodes_per_round)]
            player._rewards = rewards
            for episode in range(self._config.number_of_episodes_per_round):
                if self._games[episode].taggers[self._turn]:
                    player.total_reward_tagger += rewards[episode]
                else:
                    player.total_reward_runner += rewards[episode]
                player.memory_many[episode].experience = (choices[episode], rewards[episode], player.state_many[episode])

            # Correct total reward to average
            player.total_reward_tagger /= self._config.number_of_episodes_per_round
            player.total_reward_runner /= self._config.number_of_episodes_per_round

            # Determine if game ended and determine next state
            if i >= self._config.maximum_steps:
                for episode in range(self._config.number_of_episodes_per_round):
                    self._games[episode].ended += 1
            game_overs = [self._games[episode].ended >= self._config.number_of_players
                          for episode in range(self._config.number_of_episodes_per_round)]

            # Next turn
            self.next_turn()
            self._turn_counts = [self._turn_counts[episode] if self._games[episode].ended else self._turn_counts[episode] + 1 for episode in range(self._config.number_of_episodes_per_round)]

        print("\rPlaying over, now preparing to learn", end='')
        for player in unique_players:

            # Load all games into one memory
            for episode in range(self._config.number_of_episodes_per_round):

                # Load experiences to memory
                episode_experiences = player.memory_many[episode].experiences
                unique_episode_experiences = []
                count_end_states = 0
                for episode_experience in episode_experiences:
                    if episode_experience[3] is None:
                        count_end_states += 1
                    if (episode_experience not in unique_episode_experiences) & (count_end_states <= self._config.number_of_players):
                        unique_episode_experiences.append(episode_experience)
                        player.memory.experience = episode_experience
                        player.memory.add_experience()

            # Add rewards
            player.update_reward_store()
            player.add_rewards_to_tensorboard(np.mean(self._turn_counts))

            # Learn!
            if learn:
                if player.config.redo_batch:
                    for i in range(10):
                        print(f"\rRedoing batch, {i+1} out of 10", end='')
                        player.learn_by_replay(len(player.memory.experiences), epochs=1, verbose=True)
                else:
                    player.learn_by_replay(len(player.memory.experiences), epochs=10, verbose=True)

                # Add logs to tensorboard
                print("\rAdd logs to tensorboard", end='')
                player.add_logs_to_tensorboard()

    @property
    def players(self):
        return self._players