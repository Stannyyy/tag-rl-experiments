import pytest
from player import Player
import random
import numpy as np
import copy

@pytest.fixture
def player():
    return Player(experiment='test', name='tester')

@pytest.fixture(autouse=True)
def set_seed():
    # Set seeds before each test
    random.seed(12345)
    np.random.seed(12345)
    yield

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

