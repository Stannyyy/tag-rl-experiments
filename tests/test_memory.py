

def test_update_experience(player):
    player._state = [5,6]
    player._experience = []
    player.update_experience([0, 1, 2])
    assert player._experience == []
    player.update_experience(None)
    assert player._experience == [None, None]
    player._experience = [0, 1, 2]
    player.update_experience(None)
    assert player._experience == [0, 1, 2, None, None]
    player._experience = [0, 1, 2]
    player.update_experience([])
    assert player._experience == [0, 1, 2, [5, 6], []]
    player.update_experience([1])
    assert player._experience == [0, 1, 2, [5, 6], []]
    player._experience = [0, 1, 2]
    player.update_experience([1])
    assert player._experience == [0, 1, 2, [5, 6], [1]]


def test_add_corrected_experience(player):
    player._experience = []
    count_experiences = 0
    assert player.add_corrected_experience(True) is None
    assert player.add_corrected_experience(False) is None

    example_experience1 = [[0] * player.numStates, 1, 2, [3] * player.numStates, [4]]
    example_experience2 = [[3] * player.numStates, 0, 4, [1] * player.numStates, [2, 8]]
    example_experience3 = [[1] * player.numStates, 5, 6, [0] * player.numStates, [0, 1, 2]]

    player._experience = copy.deepcopy(example_experience1)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience2)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience3)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience1)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience2)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience3)
    player.add_corrected_experience(False);
    count_experiences += 1

    assert player._experience_buffer == [
        [[3, 3, 3, 3, 3, 3], 0, 4, [1, 1, 1, 1, 1, 1], [2, 8]],
        [[1, 1, 1, 1, 1, 1], 5, 6, [0, 0, 0, 0, 0, 0], [0, 1, 2]]]
    assert player._experience == []
    experiences = [
        [[0, 0, 0, 0, 0, 0], 1, 2, [0, 0, 0, 0, 0, 0], [0, 1, 2]],
        [[3, 3, 3, 3, 3, 3], 0, 4, [3, 3, 3, 3, 3, 3], [4]],
        [[1, 1, 1, 1, 1, 1], 5, 6, [1, 1, 1, 1, 1, 1], [2, 8]],
        [[0, 0, 0, 0, 0, 0], 1, 2, [0, 0, 0, 0, 0, 0], [0, 1, 2]]]
    assert player._experiences == experiences

    player._experience = copy.deepcopy(example_experience1)
    player.add_corrected_experience(True);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience2)
    player.add_corrected_experience(True);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience3)
    player.add_corrected_experience(True);
    count_experiences += 1

    assert player._experience_buffer == []
    assert player._experience == []
    experiences += [
        [[0, 0, 0, 0, 0, 0], 1, 2, None, None],
        [[3, 3, 3, 3, 3, 3], 0, 4, None, None],
        [[3, 3, 3, 3, 3, 3], 0, 4, [3, 3, 3, 3, 3, 3], [0]],
        [[1, 1, 1, 1, 1, 1], 5, 6, None, None],
        [[1, 1, 1, 1, 1, 1], 5, 6, [1, 1, 1, 1, 1, 1], [5]]]
    assert player._experiences == experiences

    example_experience1[3:5] = (None, None)
    player._experience = copy.deepcopy(example_experience1)
    player.add_corrected_experience(True);
    count_experiences += 1
    example_experience2[3:5] = (None, None)
    player._experience = copy.deepcopy(example_experience2)
    player.add_corrected_experience(True);
    count_experiences += 1
    example_experience3[3:5] = (None, None)
    player._experience = copy.deepcopy(example_experience3)
    player.add_corrected_experience(True);
    count_experiences += 1

    assert player._experience_buffer == []
    assert player._experience == []
    experiences += [
        [[0, 0, 0, 0, 0, 0], 1, 2, None, None],
        [[3, 3, 3, 3, 3, 3], 0, 4, None, None],
        [[1, 1, 1, 1, 1, 1], 5, 6, None, None]]
    assert player._experiences == experiences

    example_experience4 = [[2] * player.numStates, 7, 8, [2] * player.numStates, [9, 10]]
    player._experience = copy.deepcopy(example_experience4)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience4)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience4)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience4)
    player.add_corrected_experience(False);
    count_experiences += 1
    player._experience = copy.deepcopy(example_experience4)
    player.add_corrected_experience(True);
    count_experiences += 1

    assert player._experience_buffer == []
    assert player._experience == []
    experiences += [
        example_experience4,
        example_experience4,
        example_experience4,
        example_experience4,
        [[2, 2, 2, 2, 2, 2], 7, 8, None, None]]
    assert player._experiences == experiences
    assert len(player._experiences) == count_experiences


def test_add_experience(player):
    player.add_experience()
    assert player._experiences_count == 1
    assert player._experiences == [[]]
    assert player._experience == []

    player._experience = [1, 2, 3]
    player.add_experience()
    assert player._experiences_count == 2
    assert player._experiences == [[], [1, 2, 3]]
    assert player._experience == []

    player._experience = [1, 2]
    player.add_experience()
    assert player._experiences_count == 3
    assert player._experiences == [[], [1, 2, 3], [1, 2]]
    assert player._experience == []


def test_correct_experience_buffer(player):
    player._experience = [1, 2, 3]
    player._experience_buffer = [[[1, 2, 0, 1], 1, 1, [2, 1, 1, 0], []],
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
    player.correct_experience_buffer()
    assert player._experience_buffer == [[[1, 2, 2, 1], 1, 1, [2, 1, 0, 0], []]]
    assert player._experience == []
    assert player._experiences == [[[1, 2, 0, 1], 1, 1, [5, 6, 0, 1], []],
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