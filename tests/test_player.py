import pytest
import random
import numpy as np
from player import Player
from config import Config
from arena import Arena
from model import Model

@pytest.fixture
def player():
    config = Config()
    player = Player(config=config, path='tests', name='test-player')
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

def test_prediction_to_probabilities(player):

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

def test_choose_action(player):
    player._epsilon = 0.5
    player._state = [0, 1]
    if player._config.action_size != 9:
        raise Exception("Expected results for game.numActions missing")

    assert player.choose_action(options=[True, False, True, False, True, False, False, False, False],
                                save_game=False) == 4 # depends on the seed!
    assert player.choose_action(options=[True, True, False, False, False, False, False, False, True],
                                save_game=False) == 1
    assert player.choose_action(options=[False, True, True, False, False, True, False, False, False],
                                save_game=False) == 1
    assert player.choose_action(options=[False, False, True, True, True, False, False, False, False],
                                save_game=False) == 3
    assert player.choose_action(options=[False, False, False, True, True, True, True, True, True],
                                save_game=False) == 3
    assert player.choose_action(options=[False, False, False, False, True, False, False, False, False],
                                save_game=False) == 4

    assert player.choose_action([0, 2, 4], True) == 0
    assert player.choose_action([2, 3, 4], True) == 0
    assert player.choose_action([3, 4, 5, 6, 7, 8], True) == 8

    player._use_probabilities = True
    assert player.choose_action([3, 4, 5, 6, 7, 8], True) == 8
    assert player.choose_action([3, 4, 5, 6, 7, 8], False) == 8


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