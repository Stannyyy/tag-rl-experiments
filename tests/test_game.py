import pytest
from game import Game, mold_to_size
import random
import numpy as np

@pytest.fixture
def game():
    return Game(experiment='test')

@pytest.fixture(autouse=True)
def set_seed():
    # Set seeds before each test
    random.seed(12345)
    np.random.seed(12345)
    yield

def test_mold_to_size():
    assert mold_to_size(234029348092384, 5) == '23402'
    assert mold_to_size(0, 5) == '0    '
    assert mold_to_size('', 2) == '  '

def test_init_random_game(game):

    # From seed, init two random games
    if game.gridSize != 10:
        raise Exception("Expected results for game.gridSize missing")

    game.init_random_game()
    assert game._x_list == [1, 1]
    assert game._y_list == [5, 1]
    assert game._taggers == [True, False]
    assert game._ended == 0
    assert game._tag_happened == False

    game.init_random_game()
    assert game._x_list == [5, 5]
    assert game._y_list == [1, 3]
    assert game._taggers == [True, False]
    assert game._ended == 0
    assert game._tag_happened == False

def test_what_options(game):

    # For 4 corners, 4 edges and the middle, -alternating turn-, correct options
    min_i = 0
    max_i = game.gridSize - 1
    mid_i = int(game.gridSize/2)
    if game.gridSize <= 2:
        raise Exception("Expected results for game.gridSize smaller than 3 is missing")

    # Note alternating turn

    # Corner 1 - top left - can go down, right and still
    game._x_list = [min_i, mid_i]
    game._y_list = [min_i, mid_i]
    assert game.what_options(0) == [1, 3, 7, 8]

    # Corner 2 - top right - can go down, left and still
    game._x_list = [mid_i, max_i]
    game._y_list = [mid_i, min_i]
    assert game.what_options(1) == [1, 2, 6, 8]

    # Corner 3 - bottom left - can go up, right and still
    game._x_list = [min_i, mid_i]
    game._y_list = [max_i, mid_i]
    assert game.what_options(0) == [0, 3, 5, 8]

    # Corner 4 - bottom right - can go up, left and still
    game._x_list = [mid_i, max_i]
    game._y_list = [mid_i, max_i]
    assert game.what_options(1) == [0, 2, 4, 8]

    # Edge 1 - left - can go up, down, right and still
    game._x_list = [mid_i, min_i]
    game._y_list = [mid_i, mid_i]
    assert game.what_options(1) == [0, 1, 3, 5, 7, 8]

    # Edge 2 - top - can go down, right, left and still
    game._x_list = [mid_i, mid_i]
    game._y_list = [min_i, mid_i]
    assert game.what_options(0) == [1, 2, 3, 6, 7, 8]

    # Edge 3 - right - can go up, down, left and still
    game._x_list = [mid_i, max_i]
    game._y_list = [mid_i, mid_i]
    assert game.what_options(1) == [0, 1, 2, 4, 6, 8]

    # Edge 4 - bottom - can go up, right, left and still
    game._x_list = [mid_i, mid_i]
    game._y_list = [max_i, mid_i]
    assert game.what_options(0) == [0, 2, 3, 4, 5, 8]

    # Middle - can go everywhere
    game._x_list = [mid_i, mid_i]
    game._y_list = [mid_i, mid_i]
    assert game.what_options(0) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

def test_change_position(game):
    assert game.change_position(0, 1, 1) == (1, 0)
    assert game.change_position(1, 1, 1) == (1, 2)
    assert game.change_position(2, 1, 1) == (0, 1)
    assert game.change_position(3, 1, 1) == (2, 1)
    assert game.change_position(4, 1, 1) == (0, 0)
    assert game.change_position(5, 1, 1) == (2, 0)
    assert game.change_position(6, 1, 1) == (0, 2)
    assert game.change_position(7, 1, 1) == (2, 2)
    assert game.change_position(8, 1, 1) == (1, 1)

def test_move(game):
    mid_i = int(game.gridSize/2)
    if game.gridSize < 7:
        raise Exception("Expected results for game.gridSize smaller than 7 is missing")

    game._x_list = [mid_i, mid_i]
    game._y_list = [mid_i, mid_i]
    game.move(0,0)
    assert game._x_list == [mid_i, mid_i]
    assert game._y_list == [mid_i-1, mid_i]
    game.move(1,1)
    assert game._x_list == [mid_i, mid_i]
    assert game._y_list == [mid_i-1, mid_i+1]
    game.move(0,2)
    assert game._x_list == [mid_i-1, mid_i]
    assert game._y_list == [mid_i-1, mid_i+1]
    game.move(1,3)
    assert game._x_list == [mid_i-1, mid_i+1]
    assert game._y_list == [mid_i-1, mid_i+1]
    game.move(0,4)
    assert game._x_list == [mid_i-2, mid_i+1]
    assert game._y_list == [mid_i-2, mid_i+1]
    game.move(1,5)
    assert game._x_list == [mid_i-2, mid_i+2]
    assert game._y_list == [mid_i-2, mid_i]
    game.move(0,6)
    assert game._x_list == [mid_i-3, mid_i+2]
    assert game._y_list == [mid_i-1, mid_i]
    game.move(1,7)
    assert game._x_list == [mid_i-3, mid_i+3]
    assert game._y_list == [mid_i-1, mid_i+1]
    game.move(0,8)
    assert game._x_list == [mid_i-3, mid_i+3]
    assert game._y_list == [mid_i-1, mid_i+1]

def test_what_reward(game):
    mid_i = int(game.gridSize/2)
    min_i = 0
    max_i = game.gridSize - 1
    if (game.gridSize != 10) or (game.stepPoints != 1):
        raise Exception("Expected results for game.gridSize other than 10 is missing")

    game._x_list = [min_i, max_i]
    game._y_list = [min_i, max_i]
    game._taggers = [True, False]
    assert game.what_reward(0, 0) == -1 - 0.25
    assert game.what_reward(0, 1) == -1 - 0.25
    assert game.what_reward(0, 2) == -1 - 0.25
    assert game.what_reward(0, 3) == -1 - 0.25
    assert game.what_reward(0, 4) == round(-1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 5) == round(-1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 6) == round(-1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 7) == round(-1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 8) == -1
    assert game.what_reward(1, 0) == 1 - 0.25
    assert game.what_reward(1, 1) == 1 - 0.25
    assert game.what_reward(1, 2) == 1 - 0.25
    assert game.what_reward(1, 3) == 1 - 0.25
    assert game.what_reward(1, 4) == round(1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 5) == round(1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 6) == round(1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 7) == round(1 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 8) == 1

    game._x_list = [min_i, min_i]
    game._y_list = [max_i, max_i]
    game._taggers = [False, True]
    assert game.what_reward(0, 0) == -50 - 0.25
    assert game.what_reward(0, 1) == -50 - 0.25
    assert game.what_reward(0, 2) == -50 - 0.25
    assert game.what_reward(0, 3) == -50 - 0.25
    assert game.what_reward(0, 4) == round(-50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 5) == round(-50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 6) == round(-50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 7) == round(-50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(0, 8) == -50
    assert game.what_reward(1, 0) == 50 - 0.25
    assert game.what_reward(1, 1) == 50 - 0.25
    assert game.what_reward(1, 2) == 50 - 0.25
    assert game.what_reward(1, 3) == 50 - 0.25
    assert game.what_reward(1, 4) == round(50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 5) == round(50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 6) == round(50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 7) == round(50 - (0.25**2*2)**0.5, 2)
    assert game.what_reward(1, 8) == 50

def test_render(game):
    min_i = 0
    max_i = game.gridSize - 1
    mid_i = int(game.gridSize/2)
    if game.gridSize <= 2:
        raise Exception("Expected results for game.gridSize smaller than 3 is missing")

    game._x_list = [mid_i, mid_i]
    game._y_list = [mid_i, mid_i]
    game._taggers = [False, True]
    game.render()
    assert [(i,j) for j, y in enumerate(game._rendered) for i, x in enumerate(y) if x != '     '] == [(mid_i, mid_i)]
    assert game._rendered[mid_i][mid_i] == '%    '

    game._x_list = [min_i, mid_i]
    game._y_list = [mid_i, max_i]
    game._taggers = [True, False]
    game.render()
    assert [(i,j) for j, y in enumerate(game._rendered) for i, x in enumerate(y) if x != '     '] == [(min_i, mid_i), (mid_i, max_i)]
    assert game._rendered[mid_i][min_i] == 'x    '
    assert game._rendered[max_i][mid_i] == 'o    '

    game._x_list = [mid_i, min_i, mid_i, mid_i]
    game._y_list = [mid_i, max_i, max_i, max_i]
    game._taggers = [True, True, False, False]
    game.render()
    assert [(i,j) for j, y in enumerate(game._rendered) for i, x in enumerate(y) if x != '     '] == [(mid_i, mid_i), (min_i, max_i), (mid_i, max_i)]
    assert game._rendered[mid_i][mid_i] == 'x    '
    assert game._rendered[max_i][min_i] == 'x    '
    assert game._rendered[max_i][mid_i] == 'oo   '

    game._x_list = [mid_i, min_i, mid_i, mid_i]
    game._y_list = [max_i, max_i, max_i, max_i]
    game._taggers = [False, False, True, True]
    game.render()
    assert [(i,j) for j, y in enumerate(game._rendered) for i, x in enumerate(y) if x != '     '] == [(min_i, max_i), (mid_i, max_i)]
    assert game._rendered[max_i][min_i] == 'o    '
    assert game._rendered[max_i][mid_i] == '%x   '

    game._x_list = [mid_i, min_i, mid_i, mid_i]
    game._y_list = [max_i, max_i, max_i, max_i]
    game._taggers = [False, True, False, True]
    game.render()
    assert [(i,j) for j, y in enumerate(game._rendered) for i, x in enumerate(y) if x != '     '] == [(min_i, max_i), (mid_i, max_i)]
    assert game._rendered[max_i][min_i] == 'x    '
    assert game._rendered[max_i][mid_i] == 'o%   '

    if game.numActions != 9:
        raise Exception("Expected results only available for game.numActions of 9")
    game._x_list = [mid_i, mid_i]
    game._y_list = [mid_i, max_i]
    game._taggers = [True, False]
    game.render(prediction = [0.2348723, 3948.12387, 29, 1234.125, 0.0, 2394802394, 0.12, 0.2, 0], state=[None,None,None,None,1])
    assert [(i,j) for j, y in enumerate(game._rendered) for i, x in enumerate(y) if x != '     '] == [
        (mid_i, mid_i),
        (mid_i - 1, max_i - 1), (mid_i, max_i - 1), (mid_i + 1, max_i - 1),
        (mid_i - 1, max_i),     (mid_i, max_i),     (mid_i + 1, max_i)
        ]
    assert game._rendered[mid_i][mid_i] == 'x    '
    assert game._rendered[max_i][mid_i] == 'o0   '
    assert game._rendered[max_i-1][mid_i-1] == '0.0  '
    assert game._rendered[max_i-1][mid_i] == '0.234'
    assert game._rendered[max_i-1][mid_i+1] == '2e9  '
    assert game._rendered[max_i][mid_i-1] == '29   '
    assert game._rendered[max_i][mid_i+1] == '1234.'

def test_extract_position(game):
    assert game.extract_position([['','1','x'],
                                       ['.',0,0.0]], 'x') == [[0, 2]]

@pytest.mark.skip(reason="ignored: visual testing enough")
def test_draw_cat():
    pass

@pytest.mark.skip(reason="ignored: visual testing enough")
def test_draw_mouse():
    pass

@pytest.mark.skip(reason="ignored: visual testing enough")
def test_save():
    pass

@pytest.mark.skip(reason="ignored: visual testing enough")
def test_record():
    pass