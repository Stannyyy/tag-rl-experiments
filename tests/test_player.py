import pytest
import os
import random
import numpy as np
from player import Player, RandomPlayer, StillPlayer
from config import Config
from arena import Arena
from model import Model
from game import Game

@pytest.fixture
def player():
    config = Config()
    player = Player(config=config, path='tests', name='test-player')
    player.arena = Arena(config=config)
    return player

@pytest.fixture
def random_player():
    config = Config()
    player = RandomPlayer(config=config, path='tests', name='test-random-player')
    player.arena = Arena(config=config)
    return player

@pytest.fixture
def still_player():
    config = Config()
    player = StillPlayer(config=config, path='tests', name='test-still-player')
    player.arena = Arena(config=config)
    return player

@pytest.fixture
def curious_player():
    config = Config()
    config.curiosity = True
    player = Player(config=config, path='tests', name='test-player-curious')
    player.arena = Arena(config=config)
    return player

@pytest.fixture
def custom_player():
    config = Config()
    player = Player(config=config, path='', name='test-player',
                    state_size_overwrite=2, action_size_overwrite=2,
                    learning_rate = 0.001)
    player.arena = Arena(config=config)
    return player

@pytest.fixture
def game():
    config = Config()
    return Game(config)

@pytest.fixture(autouse=True)
def set_seed():
    # Set seeds before each test
    random.seed(12345)
    np.random.seed(12345)
    yield

def test_player_config():
    config = Config()
    player1 = Player(config=config, path='tests', name='test-player1')
    config.curiosity = True
    player2 = Player(config=config, path='tests', name='test-player2',
                     learning_rate=0.001,
                     game_play_mode='sequential',
                     batch_size=1000)
    assert player1.config.learning_rate != player2.config.learning_rate
    assert player1.config.curiosity != player2.config.curiosity
    assert player1.config.batch_size != player2.config.batch_size
    assert player1.config.game_play_mode != player2.config.game_play_mode

def test_prediction_to_probabilities(player, random_player, still_player):

    # Cases unispaced
    def test_cases_unispaced(player, expected_probabilities):
        probabilities = player.prediction_to_probabilities(predictions=[1, 2, 3, 4, 5],
                                                           options=[True, True, True, True, False])
        assert np.allclose(probabilities, expected_probabilities, atol=0.00001)
        probabilities = player.prediction_to_probabilities(predictions=[-2, -1, 0, 1, 2],
                                                           options=[True, True, True, True, False])
        assert np.allclose(probabilities, expected_probabilities, atol=0.00001)
        probabilities = player.prediction_to_probabilities(predictions=[[5, 6, 7, 8, 9],
                                                                        [0, 2, 4, 6, 8],
                                                                        [-100, 0, 100, 200, 300]],
                                                           options=[[True, True, True, True, False],
                                                                    [True, True, True, True, False],
                                                                    [True, True, True, True, False]])
        assert np.allclose(probabilities, np.stack([expected_probabilities]*3, axis=0), atol=0.00001)

    player._epsilon = 1
    expected_probabilities = np.array([0.24665596, 0.24887202, 0.25110798, 0.25336403, 0])
    test_cases_unispaced(player, expected_probabilities)

    player._epsilon = 0.1
    expected_probabilities = np.array([0.2175225436995875, 0.23787497331868462, 0.2601316717292149, 0.2844708112525128, 0])
    test_cases_unispaced(player, expected_probabilities)

    player._epsilon = 0.01
    expected_probabilities = np.array([0.04156005980179857, 0.10165317723042654, 0.2486369964413121, 0.6081497665264628, 0])
    test_cases_unispaced(player, expected_probabilities)

    player._epsilon = 0.001
    expected_probabilities = np.array([0.000000000002221260649791169, 0.000000017023445245169937, 0.00013046541298183262, 0.9998695175613517, 0])
    test_cases_unispaced(player, expected_probabilities)

    # Cases infinite
    def test_cases_infinite(player, expected_probabilities):
        probabilities = player.prediction_to_probabilities(predictions=[1, 2, 3, 4, -np.inf],
                                                           options=[True, True, True, True, True])
        assert np.allclose(probabilities, expected_probabilities, atol=0.00001)
        probabilities = player.prediction_to_probabilities(predictions=[-2, -1, 0, 1, -np.inf],
                                                           options=[True, True, True, True, True])
        assert np.allclose(probabilities, expected_probabilities, atol=0.00001)
        probabilities = player.prediction_to_probabilities(predictions=[[5, 6, 7, 8, -np.inf],
                                                                        [0, 2, 4, 6, -np.inf],
                                                                        [-100, 0, 100, 200, -np.inf]],
                                                           options=[[True, True, True, True, True],
                                                                    [True, True, True, True, True],
                                                                    [True, True, True, True, True]])
        assert np.allclose(probabilities, np.stack([expected_probabilities]*3, axis=0), atol=0.00001)

    player._epsilon = 1
    expected_probabilities = np.array([0.24665596478049945, 0.24887201851764362, 0.2511079821490013, 0.2533640345528555, 0])
    test_cases_infinite(player, expected_probabilities)

    expected_probabilities = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    probabilities = player.prediction_to_probabilities(predictions=[0, -np.inf, -np.inf, -np.inf, -np.inf],
                                                       options=[True, True, True, True, True])
    assert np.allclose(probabilities, expected_probabilities, atol=0.00001)


def test_state_to_prediction(custom_player):
    prediction = custom_player.state_to_prediction()
    assert prediction.shape == (0,)

    custom_player._state = [0, 1] # Don't use setter cause it's protected'
    prediction = custom_player.state_to_prediction()
    assert np.allclose(prediction, np.array([0.02986243, -0.04738425]), atol=0.00001)

    custom_player.config.add_lstm = True
    custom_player.config.sequence_length_lstm = 3
    custom_player._memory._experiences = np.array([[0, 1], [1, 0], [0, 0]])
    custom_player._model = Model(config=custom_player.config)
    prediction = custom_player.state_to_prediction()
    assert np.allclose(prediction, np.array([0.01279932, 0.02213031]), atol=0.00001)

def test_choose_action(player, random_player, still_player):
    player._epsilon = 0.5
    player._state = [0, 1]

    if player._config.action_size != 9:
        raise Exception("Expected results for game.numActions missing")

    assert player.choose_action(options=[True, False, True, False, True, False, False, False, False],
                                save_game=False) == 2 # depends on the seed!
    assert player.choose_action(options=[True, True, False, False, False, False, False, False, True],
                                save_game=False) == 1
    assert player.choose_action(options=[False, True, True, False, False, True, False, False, False],
                                save_game=False) == 5
    assert player.choose_action(options=[False, False, True, True, True, False, False, False, False],
                                save_game=False) == 4
    assert player.choose_action(options=[False, False, False, True, True, True, True, True, True],
                                save_game=False) == 8
    assert player.choose_action(options=[False, False, False, False, True, False, False, False, False],
                                save_game=False) == 4

    assert player.choose_action(options=[True, False, True, False, True, False, False, False, False],
                                save_game=True) == 0
    assert player.choose_action([False, True, True, True, False, False, False, False, False],
                                save_game=True) == 1
    assert player.choose_action([False, False, False, True, True, True, True, True, True],
                                save_game=True) == 8

    player._use_probabilities = True
    assert player.choose_action([False, False, False, True, True, True, True, True, True],
                                save_game=True) == 8
    assert player.choose_action([False, False, False, True, True, True, True, True, False],
                                save_game=False) == 3

    assert random_player.choose_action([True, True, True, True, True, True, True, True, True],
                                       save_game=True) == 3
    assert random_player.choose_action([True, True, True, True, True, True, True, True, True],
                                       save_game=False) == 5

    assert still_player.choose_action([True, True, True, True, True, True, True, True, True],
                                       save_game=True) == 8
    assert still_player.choose_action([True, True, True, True, True, True, True, True, True],
                                       save_game=False) == 8

def test_options_to_eligible_idx(player, random_player):
    options = np.array([[True, False, True, False, True, False, False, False, False],
                        [True, True, True, True, True, True, True, True, True],
                        [False, False, False, False, False, False, False, False, False],
                        [True, False, True, False, True, False, True, False, True],
                        [False, True, False, True, False, True, False, True, False]]
                       )
    option_counts, idx_options, starts = player.options_to_eligible_idx(options)
    assert np.all(option_counts == np.array([3, 9, 0, 5, 4]))
    assert np.all(idx_options == np.array([0, 2, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 4, 6, 8, 1, 3, 5, 7]))
    assert np.all(starts == np.array([0, 3, 12, 12, 17]))

    option_counts, idx_options, starts = random_player.options_to_eligible_idx(options)
    assert np.all(option_counts == np.array([3, 9, 0, 5, 4]))
    assert np.all(idx_options == np.array([0, 2, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 4, 6, 8, 1, 3, 5, 7]))
    assert np.all(starts == np.array([0, 3, 12, 12, 17]))


def test_choose_many_actions(player, random_player, still_player):
    player._epsilon = 0.5
    player._state_many = [[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1]]
    player._config.number_of_episodes_per_round = 4

    if player._config.action_size != 9:
        raise Exception("Expected results for game.numActions missing")

    options = [[True, False, True, False, True, False, False, False, False],
               [False, True, True, True, True, True, True, True, True],
               [False, False, True, True, True, True, True, True, True],
               [False, False, True, True, True, True, True, True, False]]
    assert np.all(player.choose_many_actions(options=options,
                                             selection=[0, 1, 2],
                                             competition_game=False) == [4, 6, 5, 8])

    assert np.all(player.choose_many_actions(options=options,
                                             selection=[0, 1, 2],
                                             competition_game=True) == [0, 8, 7, 8])

    player._config._use_probabilities = True
    choice = player.choose_many_actions(options=options,
                               selection=[0, 1, 2],
                               competition_game=False)
    assert np.all(choice == np.array([4, 5, 4, 8]))

    assert np.all(random_player.choose_many_actions(options=options,
                                                    selection=[0, 1, 2],
                                                    competition_game=False) == [4, 6, 5, 8])

    assert np.all(still_player.choose_many_actions(options=options,
                                                   selection=[0, 1, 2],
                                                   competition_game=False) == [8, 8, 8, 8])

def test_new_game(player, random_player, still_player):
    player._total_reward_tagger = 1
    player._total_reward_runner = 2
    player.new_game()
    assert player._total_reward_tagger == 0
    assert player._total_reward_runner == 0

    random_player._total_reward_tagger = 1
    random_player._total_reward_runner = 2
    random_player.new_game()
    assert random_player._total_reward_tagger == 0
    assert random_player._total_reward_runner == 0

    still_player._total_reward_tagger = 1
    still_player._total_reward_runner = 2
    still_player.new_game()
    assert still_player._total_reward_tagger == 0
    assert still_player._total_reward_runner == 0

@pytest.mark.skip(reason="ignored: visual testing enough")
def test_write_summary_to_tensorboard():
    pass


def test_learn_by_replay(custom_player):

    example_experience1 = [[0, 1], 0, 10, [1, 0], [True, True]]
    example_experience2 = [[0, 1], 1, -9, [1, 1], [True, True]]

    example_experience3 = [[0, 0], 0, 0, None, [False, False]]
    example_experience4 = [[0, 0], 1, 2, None, [False, False]]

    example_experience5 = [[1, 0], 0, -5, None, [False, False]]
    example_experience6 = [[1, 0], 1, 8, None, [False, False]]

    example_experience7 = [[1, 1], 0, 15, None, [False, False]]
    example_experience8 = [[1, 1], 1, -11, None, [False, False]]

    example_state1 = [0, 1]
    example_state2 = [0, 0]
    example_state3 = [1, 0]
    example_state4 = [1, 1]

    assert custom_player.learn_by_replay() == 0

    custom_player.memory._experiences += [
            example_experience1,example_experience2,example_experience3,example_experience4,
            example_experience5,example_experience6,example_experience7,example_experience8
                                          ]*100

    for i in range(100):
        assert custom_player.learn_by_replay(epochs = 1, batch_size = 800) == 1

    prediction21 = custom_player.model.predict_one([example_state1])
    prediction22 = custom_player.model.predict_one([example_state2])
    prediction23 = custom_player.model.predict_one([example_state3])
    prediction24 = custom_player.model.predict_one([example_state4])

    assert int(round(prediction21[0],0)) == 16
    assert int(round(prediction21[1],0)) == 4
    assert int(round(prediction22[0],0)) == 0
    assert int(round(prediction22[1],0)) == 2
    assert int(round(prediction23[0],0)) == -5
    assert int(round(prediction23[1],0)) == 8
    assert int(round(prediction24[0],0)) == 15
    assert int(round(prediction24[1],0)) == -11

def test_add_logs_to_tensorboard(player):
    player._epsilon = 0.28291
    player._arena._episode_count = 10
    player._arena._episode_times = [1, 3, 2]
    player._model._losses = [0.3, 0.1, 0, 1]
    player._config._curiosity = False

    q_s_a = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    q_crucial_action = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]])
    q_alternative_actions = np.array([[0, 0.1, 0.2, 0.3, 0.4]])
    player._q_logs = [[28, q_s_a, q_crucial_action, q_alternative_actions]]

    player.add_logs_to_tensorboard()
    assert player._summary_writer_collection == [{'name': 'episodes/epsilon', 'value': np.float64(0.283), 'step': 10}, {'name': 'episodes/times', 'value': np.float64(2.0), 'step': 10}, {'name': 'learning/losses', 'value': 1, 'step': 10}, {'name': 'Q/overall', 'value': np.float64(0.3), 'step': 10}, {'name': 'Q/tagged-state-of-crucial-action', 'value': np.float64(0.4), 'step': 10}, {'name': 'Q/tagged-state-of-alternative-action', 'value': np.float64(0.2), 'step': 10}]

def test_add_qs_to_tensorboard(player):
    player._arena._episode_count = 10

    q_s_a = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    q_crucial_action = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]])
    q_alternative_actions = np.array([[0, 0.1, 0.2, 0.3, 0.4]])
    player._q_logs = [[28, q_s_a, q_crucial_action, q_alternative_actions]]

    player.add_qs_to_tensorboard()
    part1 = [{'name': 'Q/overall', 'value': np.float64(0.3), 'step': 10}, {'name': 'Q/tagged-state-of-crucial-action', 'value': np.float64(0.4), 'step': 10}, {'name': 'Q/tagged-state-of-alternative-action', 'value': np.float64(0.2), 'step': 10}]
    assert player._summary_writer_collection == part1

    player._q_logs = [[0, q_s_a, q_crucial_action, q_alternative_actions]]
    player.add_qs_to_tensorboard()
    part2 = [{'name': 'Q/overall', 'value': np.float64(0.3), 'step': 10}]
    assert player._summary_writer_collection == part1 + part2

    player._q_logs = [[0, q_s_a]]
    player.add_qs_to_tensorboard()
    part3 = [{'name': 'Q/overall', 'value': np.float64(0.3), 'step': 10}]
    assert player._summary_writer_collection == part1 + part2 + part3

def test_add_losses_to_tensorboard(player, curious_player):
    player._arena._episode_count = 10
    player._model._losses = [0.3, 0.1, 0, 1]
    player.add_losses_to_tensorboard()
    assert player._summary_writer_collection == [{'name': 'learning/losses', 'value': 1, 'step': 10}]

    curious_player._arena._episode_count = 10
    curious_player._model._losses = [0.3, 0.1, 0, 1]
    curious_player._model_next_state._losses_next_state = [0.3, 0.1, 0, 1, 8]
    curious_player.add_losses_to_tensorboard()
    assert curious_player._summary_writer_collection == [{'name': 'learning/losses', 'value': 1, 'step': 10}, {'name': 'learning/losses-next-state', 'value': 8, 'step': 10}]

def test_add_epsilon_to_tensorboard(player):
    player._arena._episode_count = 10
    player._epsilon = 0.28291
    player.add_epsilon_to_tensorboard()
    assert player._summary_writer_collection == [{'name': 'episodes/epsilon', 'value': np.float64(0.283), 'step': 10}]

def test_add_episode_times_to_tensorboard(player):
    player._arena._episode_count = 10
    player._arena._episode_times = [1, 3, 2]
    player.add_episode_times_to_tensorboard()
    assert player._summary_writer_collection == [{'name': 'episodes/times', 'value': np.float64(2.0), 'step': 10}]

def test_add_rewards_to_tensorboard(player):
    player._arena._episode_count = 10
    player.add_rewards_to_tensorboard(28)
    part1 = [{'name': 'rewards/turn_count', 'value': 28, 'step': 10}]
    assert player._summary_writer_collection == part1

    player._total_reward_runner = 30
    player.add_rewards_to_tensorboard(45)
    part2 = [{'name': 'rewards/turn_count', 'value': 45, 'step': 10}, {'name': 'rewards/runner', 'value': 30.0, 'step': 10}]
    assert player._summary_writer_collection == part1 + part2

    player._total_reward_tagger = 49
    player.add_rewards_to_tensorboard(60)
    part3 = [{'name': 'rewards/turn_count', 'value': 60, 'step': 10}, {'name': 'rewards/tagger', 'value': 49.0, 'step': 10}, {'name': 'rewards/runner', 'value': 30.0, 'step': 10}]
    assert player._summary_writer_collection == part1 + part2 + part3

def test_add_competition_to_tensorboard(player):
    player._total_reward_runner = 30
    player._total_reward_tagger = 49
    player.add_competition_to_tensorboard(28, 'tester')
    assert player._summary_writer_collection == [{'name': 'competition/tester/tagger', 'value': 49.0, 'step': 28}, {'name': 'competition/tester/runner', 'value': 30.0, 'step': 28}]

def test_create_batch(custom_player):
    custom_player._memory._experiences = np.array([[0, 1], [1, 0], [0, 0]])
    batch = custom_player.create_batch()
    assert batch == []

    custom_player._config._batch_size = 2
    batch = custom_player.create_batch()
    assert np.all(batch[0][0] == np.array([0, 0]))
    assert np.all(batch[1][0] == np.array([0, 1]))

    batch = custom_player.create_batch(1)
    assert np.all(batch[0][0] == np.array([1, 0]))

    # TODO: add case LSTM
    custom_player._config._add_lstm = 10
    custom_player._model._sequence_length_lstm = 2

def test_show_q_in_state(player, game):
    player._state = [0, 1, 0, 1, 0, 1]
    to_print = player.show_q_in_state(game)
    assert to_print == '\n---\n[0, 1, 0, 1, 0, 1]\n---\n     |     |     |     |     |     |     |     |     |     \n     |     |     |o    |     |     |     |-0.1 |0.3  |0.1  \n     |     |     |     |     |     |     |-0.2 |x0.1 |-0.0 \n     |     |     |     |     |     |     |-0.0 |-0.0 |0.1  \n     |     |     |     |     |     |     |     |     |     \n     |     |     |     |     |     |     |     |     |     \n     |     |     |     |     |     |     |     |     |     \n     |     |     |     |     |     |     |     |     |     \n     |     |     |     |     |     |     |     |     |     \n     |     |     |     |     |     |     |     |     |     \n---'

def test_update_reward_store(player):
    player.update_reward_store()
    assert player._reward_store_tagger == []
    assert player._reward_store_runner == []

    player._total_reward_tagger = 1
    player.update_reward_store()
    assert player._reward_store_tagger == [1]
    assert player._reward_store_runner == []

    player._total_reward_runner = 2
    player.update_reward_store()
    assert player._reward_store_tagger == [1, 1]
    assert player._reward_store_runner == [2]

def test_update_epsilon(player):
    player._epsilon = 0.5
    player.update_epsilon()
    assert np.round(player._epsilon, 3) == 1.000

    player._config._minimum_epsilon = 0.01
    player._config._maximum_epsilon = 0.90
    player._config._bootstrap_value_epsilon = 0.0001
    player._arena._episode_count = 1000
    player.update_epsilon()
    assert np.round(player._epsilon, 3) == 0.815

    player._arena._episode_count = 10000
    player.update_epsilon()
    assert np.round(player._epsilon, 3) == 0.337

    player._arena._episode_count = 100000
    player.update_epsilon()
    assert np.round(player._epsilon, 3) == 0.01

def test_reload(player):
    input_path = os.getcwd() + r'\tests\input\test.weights.h5'
    player.reload(player._arena, input_path)
    player._state = [1, 1, 1, 2, 0, 1] # Player is tagger at spot 0, so go down (action 1)
    assert player.choose_action(options=[True, True, True, True, True, True, True, True, True],
                                save_game=True) == 1

    player._state = [1, 2, 1, 1, 0, 1] # Player is tagger at spot 0, so go right (action 3)
    assert player.choose_action(options=[True, True, True, True, True, True, True, True, True],
                                save_game=True) == 3

## TODO Random en Still players