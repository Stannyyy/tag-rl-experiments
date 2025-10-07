# Import packages
import numpy as np
import pickle
import datetime
import os

# Arena
class Arena:
    def __init__(self, config, moderator=None):
        
        self._config = config
        self._episode_count = 1
        self._start_time = datetime.datetime.now()
        self._episode_times = []
        self._moderator = moderator

    def play_and_learn(self):

        new_round = True
        # Loop for number of episodes
        if self._episode_count < self._config.number_of_episodes_total:
            self.start_stopwatch()
            if self._config.game_play_mode == 'sequential':
                while (self._episode_count % self._config.number_of_episodes_per_round != 1) | (new_round == True):
                    new_round = False

                    # Play episode!
                    self._moderator.play_one(False)
                    if len(self._moderator.players[0].losses) > 0:
                        self.progress_bar(task=f'Playing episode: {self._episode_count} with loss {str(np.round(self._moderator.players[0].model.losses[-1],2))}')
                    self._episode_count += 1

                    # Stop the stopwatch
                    self.stop_stopwatch()
            elif self._config.game_play_mode == 'parallel':
                self._moderator.play_many()
                self.stop_stopwatch()
                self._episode_count += self._config.number_of_episodes_per_round

            # Print progress
            print(f'\nRound {self._episode_count-1} out of {self._config.number_of_episodes_total} - {self._episode_times[-1]} seconds elapsed')

            unique_players = list(set(self._moderator.players))
            for player in unique_players:
                if ~player.is_random:
                    # Print progress
                    av_loss = np.array(player.model.losses[-100:]).mean().round(3)
                    av_rwd_tagger = np.array(player.reward_store_tagger[-100:]).mean().round(2)
                    av_rwd_runner = np.array(player.reward_store_runner[-100:]).mean().round(2)
                    epsilon = round(player.epsilon, 2)
                    print(player.name + '; average loss: ' + str(av_loss) + ', epsilon: ' + str(epsilon) + ', average reward tagger: ' + str(
                        av_rwd_tagger) + ', average reward runner: ' + str(av_rwd_runner))

                    # Check if learning done
                    if av_loss < 0.0001:
                        self._episode_count = self._config.number_of_episodes_total # Call it a day

            # Show a couple of episodes
            if self._config.create_video:
                for i in range(2):
                    self._moderator.play_one(save_game=True, learn=False)

            # Save models
            self.save_status()

    def save_status(self):
        unique_players = list(set(self._moderator.players))
        for player in unique_players:
            if player.is_random == False:

                # Save model
                if self._config.test_mode == False:
                    checkpoint_path_version = os.path.join(player.checkpoint_path, f'cp-{self._episode_count:06d}')
                    player.model.save_checkpoint(checkpoint_path_version)
                    if player.config.curiosity:
                        player.model.save_checkpoint_next_state(checkpoint_path_version)
                player.write_summary_to_tensorboard()

                # Save status
                self._moderator = None
                with open(player.state_path, "wb") as file_:
                    pickle.dump(self, file_, -1)
                player.reload(self)


    def competition(self):

        # Reload players
        for player in self._moderator.players:
            player.reload(self)

        # Loop for number of episodes
        match_count = 0
        while match_count < 10:
            match_count += 1
            print(f"\rShowing {match_count} out of {10} before starting large competition", end='')
            self._moderator.play_one(save_game=True, learn=False)

        self._moderator.play_many(learn=False)

    def start_stopwatch(self):
        self._start_time = datetime.datetime.now()

    def stop_stopwatch(self):
        self._episode_times += [(datetime.datetime.now() - self._start_time).seconds]

    def progress_bar(self, task, based_on='episodes', i=100, total=None):
        if based_on == 'episodes':
            total = self._config.number_of_episodes_per_round
            i = self._episode_count
        elif based_on == 'i':
            if total is None:
                total = self._config.number_of_episodes_per_round / 10
        percent = int(np.ceil((100 * (i % total / float(total)))))
        if percent == 0:
            percent = 100
        bar = '█' * percent + '-' * (100 - percent)
        print(f"\r|{bar}| {percent}%   {task}  ", end="")

    @property
    def episode_count(self):
        return self._episode_count

    @property
    def episode_times(self):
        return self._episode_times

    @episode_times.setter
    def episode_times(self, episode_times):
        self._episode_times = episode_times

    @property
    def moderator(self):
        return self._moderator

    @moderator.setter
    def moderator(self, moderator):
        self._moderator = moderator
