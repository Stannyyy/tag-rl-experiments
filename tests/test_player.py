import pytest
from player import Player
import random
import numpy as np
import copy

@pytest.fixture
def player():
    return Player(experiment='test', name='tester')

@pytest.fixture
def custom_player():
    return Player(experiment='test', name='tester',
                  numStatesOverwrite=2, numActionsOverwrite=2)

@pytest.fixture(autouse=True)
def set_seed():
    # Set seeds before each test
    random.seed(12345)
    np.random.seed(12345)
    yield

def test_prediction_to_probabilities(player):
    player._eps = 1
    four_uni_spaced = np.array([0.24665596478049945, 0.24887201851764362, 0.2511079821490013, 0.2533640345528555])
    probabilities = player.prediction_to_probabilities([0, 1, 2, 3])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-2, -1, 0, 1])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([5, 6, 7, 8])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([0, 2, 4, 6])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-100, 0, 100, 200])
    assert np.all(probabilities == four_uni_spaced)

    player._eps = 0.1
    four_uni_spaced = np.array([0.2175225436995875, 0.23787497331868462, 0.2601316717292149, 0.2844708112525128])
    probabilities = player.prediction_to_probabilities([0, 1, 2, 3])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-2, -1, 0, 1])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([5, 6, 7, 8])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([0, 2, 4, 6])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-100, 0, 100, 200])
    assert np.all(probabilities == four_uni_spaced)

    player._eps = 0.01
    four_uni_spaced = np.array([0.04156005980179857, 0.10165317723042654, 0.2486369964413121, 0.6081497665264628])
    probabilities = player.prediction_to_probabilities([0, 1, 2, 3])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-2, -1, 0, 1])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([5, 6, 7, 8])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([0, 2, 4, 6])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-100, 0, 100, 200])
    assert np.all(probabilities == four_uni_spaced)

    player._eps = 0.001
    four_uni_spaced = np.array([0.000000000002221260649791169, 0.000000017023445245169937, 0.00013046541298183262, 0.9998695175613517])
    probabilities = player.prediction_to_probabilities([0, 1, 2, 3])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-2, -1, 0, 1])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([5, 6, 7, 8])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([0, 2, 4, 6])
    assert np.all(probabilities == four_uni_spaced)
    probabilities = player.prediction_to_probabilities([-100, 0, 100, 200])
    assert np.all(probabilities == four_uni_spaced)

    player._eps = 1
    five_inf = np.array([0.24665596478049945, 0.24887201851764362, 0.2511079821490013, 0.2533640345528555, 0])
    probabilities = player.prediction_to_probabilities([0, 1, 2, 3, -np.inf])
    assert np.all(probabilities == five_inf)
    probabilities = player.prediction_to_probabilities([-2, -1, 0, 1, -np.inf])
    assert np.all(probabilities == five_inf)
    probabilities = player.prediction_to_probabilities([5, 6, 7, 8, -np.inf])
    assert np.all(probabilities == five_inf)
    probabilities = player.prediction_to_probabilities([0, 2, 4, 6, -np.inf])
    assert np.all(probabilities == five_inf)
    probabilities = player.prediction_to_probabilities([-100, 0, 100, 200, -np.inf])
    assert np.all(probabilities == five_inf)

    five_inf = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0])
    probabilities = player.prediction_to_probabilities([0, -np.inf, -np.inf, -np.inf, -np.inf])
    assert np.all(probabilities == five_inf)


def test_choose_action(player):
    player._eps = 0.5
    player._state = [0, 1]
    if player.numActions != 9:
        raise Exception("Expected results for game.numActions missing")

    assert player.choose_action([0, 2, 4], False) == 0 # depends on the seed!
    assert player.choose_action([0, 1, 8], False) == 0
    assert player.choose_action([1, 2, 5], False) == 2
    assert player.choose_action([2, 3, 4], False) == 3
    assert player.choose_action([3, 4, 5, 6, 7, 8], False) == 3
    assert player.choose_action([4], False) == 4

    assert player.choose_action([0, 2, 4], True) == 2
    assert player.choose_action([2, 3, 4], True) == 2
    assert player.choose_action([3, 4, 5, 6, 7, 8], True) == 7

    player._use_probabilities = True
    assert player.choose_action([3, 4, 5, 6, 7, 8], True) == 7
    assert player.choose_action([3, 4, 5, 6, 7, 8], False) == 8

@pytest.mark.skip(reason="ignored: visual testing enough")
def test_write_summary_to_tensorboard():
    pass

def test_state_set_and_get(player):
    game_x = [1, 2]
    game_y = [3, 4]
    turn = 5
    is_tagger = True
    player.state = (game_x, game_y, turn, is_tagger)
    expected = game_x + game_y + [turn, 1]
    assert player.state == expected

def test_sample_set_and_get(player):
    player._state = [1, 2, 3, 4, 5, 1]
    player.sample = (0, 0.2347)
    assert player.sample == [[1, 2, 3, 4, 5, 1], 0, 0.2347]

def test_update_sample(player):
    player._state = [5,6]
    player._sample = []
    player.update_sample([0, 1, 2])
    assert player._sample == []
    player.update_sample(None)
    assert player._sample == [None, None]
    player._sample = [0, 1, 2]
    player.update_sample(None)
    assert player._sample == [0, 1, 2, None, None]
    player._sample = [0, 1, 2]
    player.update_sample([])
    assert player._sample == [0, 1, 2, [5, 6], []]
    player.update_sample([1])
    assert player._sample == [0, 1, 2, [5, 6], []]
    player._sample = [0, 1, 2]
    player.update_sample([1])
    assert player._sample == [0, 1, 2, [5, 6], [1]]

def test_add_corrected_sample(player):
    player._sample = []
    count_samples = 0
    assert player.add_corrected_sample(True) is None
    assert player.add_corrected_sample(False) is None

    example_sample1 = [[0] * player.numStates, 1, 2, [3] * player.numStates, [4]]
    example_sample2 = [[3] * player.numStates, 0, 4, [1] * player.numStates, [2,8]]
    example_sample3 = [[1] * player.numStates, 5, 6, [0] * player.numStates, [0,1,2]]

    player._sample = copy.deepcopy(example_sample1)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample2)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample3)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample1)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample2)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample3)
    player.add_corrected_sample(False); count_samples += 1
    
    assert player._sample_buffer == [
        [[3, 3, 3, 3, 3, 3], 0, 4, [1, 1, 1, 1, 1, 1], [2, 8]],
        [[1, 1, 1, 1, 1, 1], 5, 6, [0, 0, 0, 0, 0, 0], [0, 1, 2]]]
    assert player._sample == []
    samples = [
        [[0, 0, 0, 0, 0, 0], 1, 2, [0, 0, 0, 0, 0, 0], [0, 1, 2]],
        [[3, 3, 3, 3, 3, 3], 0, 4, [3, 3, 3, 3, 3, 3], [4]],
        [[1, 1, 1, 1, 1, 1], 5, 6, [1, 1, 1, 1, 1, 1], [2, 8]],
        [[0, 0, 0, 0, 0, 0], 1, 2, [0, 0, 0, 0, 0, 0], [0, 1, 2]]]
    assert player._samples == samples

    player._sample = copy.deepcopy(example_sample1)
    player.add_corrected_sample(True); count_samples += 1
    player._sample = copy.deepcopy(example_sample2)
    player.add_corrected_sample(True); count_samples += 1
    player._sample = copy.deepcopy(example_sample3)
    player.add_corrected_sample(True); count_samples += 1

    assert player._sample_buffer == []
    assert player._sample == []
    samples += [
        [[0, 0, 0, 0, 0, 0], 1, 2, None, None],
        [[3, 3, 3, 3, 3, 3], 0, 4, None, None],
        [[3, 3, 3, 3, 3, 3], 0, 4, [3, 3, 3, 3, 3, 3], [0]],
        [[1, 1, 1, 1, 1, 1], 5, 6, None, None],
        [[1, 1, 1, 1, 1, 1], 5, 6, [1, 1, 1, 1, 1, 1], [5]]]
    assert player._samples == samples

    example_sample1[3:5] = (None, None)
    player._sample = copy.deepcopy(example_sample1)
    player.add_corrected_sample(True); count_samples += 1
    example_sample2[3:5] = (None, None)
    player._sample = copy.deepcopy(example_sample2)
    player.add_corrected_sample(True); count_samples += 1
    example_sample3[3:5] = (None, None)
    player._sample = copy.deepcopy(example_sample3)
    player.add_corrected_sample(True); count_samples += 1

    assert player._sample_buffer == []
    assert player._sample == []
    samples += [
        [[0, 0, 0, 0, 0, 0], 1, 2, None, None],
        [[3, 3, 3, 3, 3, 3], 0, 4, None, None],
        [[1, 1, 1, 1, 1, 1], 5, 6, None, None]]
    assert player._samples == samples

    example_sample4 = [[2] * player.numStates, 7, 8, [2] * player.numStates, [9,10]]
    player._sample = copy.deepcopy(example_sample4)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample4)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample4)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample4)
    player.add_corrected_sample(False); count_samples += 1
    player._sample = copy.deepcopy(example_sample4)
    player.add_corrected_sample(True); count_samples += 1

    assert player._sample_buffer == []
    assert player._sample == []
    samples += [
        example_sample4,
        example_sample4,
        example_sample4,
        example_sample4,
        [[2, 2, 2, 2, 2, 2], 7, 8, None, None]]
    assert player._samples == samples
    assert len(player._samples) == count_samples

def test_add_sample(player):
    player.add_sample()
    assert player._samples_count == 1
    assert player._samples == [[]]
    assert player._sample == []

    player._sample = [1, 2, 3]
    player.add_sample()
    assert player._samples_count == 2
    assert player._samples == [[], [1, 2, 3]]
    assert player._sample == []

    player._sample = [1, 2]
    player.add_sample()
    assert player._samples_count == 3
    assert player._samples == [[], [1, 2, 3], [1, 2]]
    assert player._sample == []

def test_correct_sample_buffer(player):
    player._sample = [1, 2, 3]
    player._sample_buffer = [[[1, 2, 0, 1], 1, 1, [2, 1, 1, 0], []],
                             [[3, 4, 1, 0], 1, 1, [5, 6, 0, 1], []],
                             [[7, 8, 1, 0], 1, 1, [9, 10, 0, 0], []],
                             [[0, 1, 0, 0], 1, 1, [2, 3, 1, 1], []],
                             [[4, 5, 1, 1], 1, 1, [6, 7, 0, 0], []],
                             [[8, 9, 0, 1], 1, 1, [10, 0, 1, 0], []],
                             [[1, 2, 0, 0], 1, 1, [3, 4, 1, 1], []],
                             [[5, 6, 1, 0], 1, 1, [7, 8, 1, 0], []],
                             [[1, 2, 0, 1], 1, 1, [2, 1, 0, 0], []],
                             [[1, 2, 2, 1], 1, 1, [2, 1, 0, 0], []],
                             [[1, 2, 0, 1], 1, 1, [8, 1, 2, 0], []],
                             [[1, 2, 2, 1], 1, 1, [2, 1, 0, 0], []],
                             [[9, 10, 1, 0], 1, 1, None, None],
                             [[0, 1, 0, 1], 1, 1, None, None]]
    player.correct_sample_buffer()
    assert player._sample_buffer == [[[1, 2, 2, 1], 1, 1, [2, 1, 0, 0], []]]
    assert player._sample == []
    assert player._samples == [[[1, 2, 0, 1], 1, 1, [5, 6, 0, 1], []],
                               [[3, 4, 1, 0], 1, 1, [2, 3, 1, 1], []],
                               [[7, 8, 1, 0], 1, 1, [2, 3, 1, 1], []],
                               [[0, 1, 0, 0], 1, 1, [6, 7, 0, 0], []],
                               [[4, 5, 1, 1], 1, 1, [10, 0, 1, 0], []],
                               [[8, 9, 0, 1], 1, 1, [2, 1, 0, 0], []],
                               [[1, 2, 0, 0], 1, 1, [2, 1, 0, 0], []],
                               [[5, 6, 1, 0], 1, 1, [9, 10, 1, 0], [1]],
                               [[1, 2, 0, 1], 1, 1, [2, 1, 0, 0], []],
                               [[1, 2, 2, 1], 1, 1, [8, 1, 2, 0], []],
                               [[1, 2, 0, 1], 1, 1, [2, 1, 0, 0], []]]

def test_learn_by_replay(custom_player):

    example_sample1 = [[0, 1], 0, 10, [1, 0], [0, 1]]
    example_sample2 = [[0, 1], 1, -9, [1, 1], [0, 1]]

    example_sample3 = [[0, 0], 0, 0, None, None]
    example_sample4 = [[0, 0], 1, 2, None, None]

    example_sample5 = [[1, 0], 0, -5, None, None]
    example_sample6 = [[1, 0], 1, 8, None, None]

    example_sample7 = [[1, 1], 0, 15, None, None]
    example_sample8 = [[1, 1], 1, -11, None, None]

    example_state1 = [0, 1]
    example_state2 = [0, 0]
    example_state3 = [1, 0]
    example_state4 = [1, 1]

    prediction11 = custom_player.predict_one([example_state1])
    prediction12 = custom_player.predict_one([example_state2])
    prediction13 = custom_player.predict_one([example_state3])
    prediction14 = custom_player.predict_one([example_state4])

    assert all(np.abs(np.round(prediction11,1)) <= 0.1)
    assert all(np.abs(np.round(prediction12, 1)) <= 0.1)
    assert all(np.abs(np.round(prediction13, 1)) <= 0.1)
    assert all(np.abs(np.round(prediction14, 1)) <= 0.1)

    assert custom_player.learn_by_replay() == 0
    for i in range(100):
        custom_player.samples += [example_sample1]
        custom_player.samples += [example_sample2]
        custom_player.samples += [example_sample3]
        custom_player.samples += [example_sample4]
        custom_player.samples += [example_sample5]
        custom_player.samples += [example_sample6]
        custom_player.samples += [example_sample7]
        custom_player.samples += [example_sample8]

    for i in range(100):
        assert custom_player.learn_by_replay() == 1

    prediction21 = custom_player.predict_one([example_state1])
    prediction22 = custom_player.predict_one([example_state2])
    prediction23 = custom_player.predict_one([example_state3])
    prediction24 = custom_player.predict_one([example_state4])

    assert int(round(prediction21[0],0)) == 16
    assert int(round(prediction21[1],0)) == 4
    assert int(round(prediction22[0],0)) == 0
    assert int(round(prediction22[1],0)) == 2
    assert int(round(prediction23[0],0)) == -5
    assert int(round(prediction23[1],0)) == 8
    assert int(round(prediction24[0],0)) == 15
    assert int(round(prediction24[1],0)) == -11