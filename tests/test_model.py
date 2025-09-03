import os
import numpy as np
import pytest
from model import Model, ModelNextState
import random

@pytest.fixture
def models():
    return (Model(experiment='tests'),
            Model(experiment='tests', addLSTM=True, sequenceLengthLSTM=3),
            Model(experiment='tests', layers=[10, 10, 10]),
            Model(experiment='tests', addLSTM=True, sequenceLengthLSTM=3, layers=[10, 10, 10]),
            Model(experiment='tests', learningRate=0.01),)

@pytest.fixture
def models_next_state():
    return (ModelNextState(experiment='tests'),
            ModelNextState(experiment='tests', addLSTM=True, sequenceLengthLSTM=3))

@pytest.fixture(autouse=True)
def set_seed():
    # Set seeds before each test
    random.seed(12345)
    np.random.seed(12345)
    yield

def test_define_model_predict_one(models):
    model1, model2, model3, model4, _ = models

    pred1 = model1.predict_one(state=[[0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (model1.numActions,)
    assert len(model1._model.layers) == len(model1._layers)*2+1

    state = [list(range(model2.numStates))]
    samples = [[state[0],0]] * 10
    last_x_minus_1_samples = samples[(model2._sequence_length_LSTM * -1 + 1):]
    pred2 = model2.predict_one(state=[s[0] for s in last_x_minus_1_samples] + state)
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2.numActions,)
    assert len(model2._model.layers) == ((len(model2._layers)-1)*2)+2

    pred3 = model3.predict_one(state=[[0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred3, np.ndarray)
    assert pred3.shape == (model3.numActions,)
    assert len(model3._model.layers) == len(model3._layers)*2+1

    state = [list(range(model4.numStates))]
    samples = [[state[0],0]] * 10
    last_x_minus_1_samples = samples[(model2._sequence_length_LSTM * -1 + 1):]
    pred4 = model4.predict_one(state=[s[0] for s in last_x_minus_1_samples] + state)
    assert isinstance(pred4, np.ndarray)
    assert pred4.shape == (model4.numActions,)
    assert len(model4._model.layers) == ((len(model4._layers)-1)*2)+2


def test_predict_batch(models):
    model1, model2, model3, model4, _ = models

    pred1 = model1.predict_batch(states=[[0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (3, model1.numActions)
    assert len(model1._model.layers) == len(model1._layers)*2+1

    states = [[list(range(model2.numStates))]*3]*model2.batchSize
    pred2 = model2.predict_batch(states=np.array(states))
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2.batchSize,model2.numActions)
    assert len(model2._model.layers) == ((len(model2._layers)-1)*2)+2

    pred3 = model3.predict_batch(states=[[0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred3, np.ndarray)
    assert pred3.shape == (3, model1.numActions)
    assert len(model3._model.layers) == len(model3._layers)*2+1

    states = [[list(range(model4.numStates))]*3]*model4.batchSize
    pred4 = model4.predict_batch(states=np.array(states))
    assert isinstance(pred4, np.ndarray)
    assert pred4.shape == (model4.batchSize,model4.numActions)
    assert len(model4._model.layers) == ((len(model4._layers)-1)*2)+2


def test_train_batch(models):
    model1, model2, model3, model4, model5 = models

    # Case 1
    x_batch = np.array([list(range(model1.numStates))]*model1.batchSize)
    y_batch = np.array([list(range(model1.numActions))]*model1.batchSize)

    prediction1 = model1.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    tb = model1.train_batch(x_batch=x_batch, y_batch=y_batch, step=1)
    assert tb == {'name': 'params/losses', 'value': 23.5631, 'step': 1}
    assert model1._losses == [23.56310272216797]

    prediction2 = model1.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == 0.00640537188287323

    # Case 2
    x_batch = np.array([[list(range(model2.numStates))]*3]*model2.batchSize).astype(float)
    y_batch = np.array([list(range(model2.numActions))]*model2.batchSize).astype(float)

    prediction1 = model2.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    tb = model2.train_batch(x_batch=x_batch, y_batch=y_batch, step=2)
    assert tb == {'name': 'params/losses', 'value': 22.39533, 'step': 2}
    assert model2._losses == [22.395326614379883]

    prediction2 = model2.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == 0.004100450231961105

    # Case 3
    x_batch = np.array([list(range(model3.numStates))]*model3.batchSize)
    y_batch = np.array([list(range(model3.numActions))]*model3.batchSize)

    prediction1 = model3.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    tb = model3.train_batch(x_batch=x_batch, y_batch=y_batch, step=3)
    assert tb == {'name': 'params/losses', 'value': 28.58597, 'step': 3}
    assert model3._losses == [28.585973739624023]

    prediction2 = model3.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == 0.0017196189773672856

    # Case 4
    x_batch = np.array([[list(range(model2.numStates))]*3]*model2.batchSize).astype(float)
    y_batch = np.array([list(range(model2.numActions))]*model2.batchSize).astype(float)

    prediction1 = model4.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    tb = model4.train_batch(x_batch=x_batch, y_batch=y_batch, step=4)
    assert tb == {'name': 'params/losses', 'value': 22.56413, 'step': 4}
    assert model4._losses == [22.564132690429688]

    prediction2 = model4.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == 0.0007378582012387377

    # Case 5
    x_batch = np.array([list(range(model5.numStates))]*model5.batchSize)
    y_batch = np.array([list(range(model5.numActions))]*model5.batchSize)

    prediction1 = model5.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    tb = model5.train_batch(x_batch=x_batch, y_batch=y_batch, step=5)
    assert tb == {'name': 'params/losses', 'value': 21.10648, 'step': 5}
    assert model5._losses == [21.106477737426758]

    prediction2 = model5.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert round((prediction1_err-prediction2_err)/prediction1_err,3) == 0.400


def test_save_checkpoint(models):
    model1, _, _, _, _ = models
    x_batch = np.array([list(range(model1.numStates))]*model1.batchSize)
    model1.predict_batch(states=x_batch)
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1'), exist_ok=True)
    model1.save_checkpoint(1, 'testplayer', 'phase1')
    assert 'cp-000001.weights.h5' in os.listdir(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1'))


def test_load_checkpoint(models):
    model1, _, _, _, _ = models
    x_batch = np.array([list(range(model1.numStates))]*model1.batchSize)
    model1.predict_batch(states=x_batch)
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1'), exist_ok=True)
    weights_before = model1._model.weights
    model1.save_checkpoint(1, 'testplayer', 'phase1')
    model1._model.load_weights(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1', 'cp-000001.weights.h5'))
    weights_after = model1._model.weights
    assert weights_before == weights_after


def test_define_model_predict_one_next_state(models_next_state):
    model1, model2 = models_next_state

    pred1 = model1.predict_one_next_state(state=[[0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (model1.numStates,)
    assert len(model1._model_next_state.layers) == len(model1._layers_next_state) * 2 + 1

    state = [list(range(model2.numStates))]
    samples = [[state[0], 0]] * 10
    last_x_minus_1_samples = samples[(model2._sequence_length_LSTM * -1 + 1):]
    pred2 = model2.predict_one_next_state(state=[s[0] for s in last_x_minus_1_samples] + state)
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2._sequence_length_LSTM, model2.numStates)
    assert len(model2._model_next_state.layers) == (len(model2._layers_next_state) * 2) + 1


def test_predict_batch_next_state(models_next_state):
    model1, model2 = models_next_state

    pred1 = model1.predict_batch_next_state(states=[[0.0, 1.0, 2.0, 3.0],
                                                    [0.0, 1.0, 2.0, 3.0],
                                                    [0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (3, model1.numStates)
    assert len(model1._model_next_state.layers) == len(model1._layers_next_state) * 2 + 1

    states = [[list(range(model2.numStates))] * model2._sequence_length_LSTM] * model2.batchSize
    pred2 = model2.predict_batch_next_state(states=np.array(states))
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2.batchSize, model2._sequence_length_LSTM, model2.numStates)
    assert len(model2._model_next_state.layers) == (len(model2._layers_next_state) * 2) + 1


def test_train_batch_next_state(models_next_state):
    model1, model2 = models_next_state

    # Case 1
    x_batch = np.array([list(range(model1.numStates))] * model1.batchSize)
    y_batch = np.array([list(range(model1.numStates))] * model1.batchSize)

    prediction1 = model1.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch - prediction1))

    tb = model1.train_batch_next_state(x_batch=x_batch, y_batch=y_batch, step=1)
    assert tb == {'name': 'params/losses-next-state', 'value': 6.71815, 'step': 1}
    assert model1._losses_next_state == [6.7181477546691895]

    prediction2 = model1.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch - prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err - prediction2_err) / prediction1_err == 0.10701045583093437

    # Case 2
    x_batch = np.array([[list(range(model2.numStates))] * 3] * model2.batchSize).astype(float)
    y_batch = np.array([list(range(model2.numStates))] * model2.batchSize).astype(float)

    prediction1 = model2.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch - prediction1))

    tb = model2.train_batch_next_state(x_batch=x_batch, y_batch=y_batch, step=2)
    assert tb == {'name': 'params/losses-next-state', 'value': 9.25949, 'step': 2}
    assert model2._losses_next_state == [9.259491920471191]

    prediction2 = model2.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch - prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err - prediction2_err) / prediction1_err == 0.07853682469434878


def test_save_checkpoint_next_state(models_next_state):
    model1, _ = models_next_state
    x_batch = np.array([list(range(model1.numStates))] * model1.batchSize)
    model1.predict_batch_next_state(states=x_batch)
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1'), exist_ok=True)
    model1.save_checkpoint_next_state(1, 'testplayer', 'phase1')
    assert 'cp-000001-next-state.weights.h5' in os.listdir(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1'))


def test_load_checkpoint_next_state(models_next_state):
    model1, _ = models_next_state
    x_batch = np.array([list(range(model1.numStates))] * model1.batchSize)
    model1.predict_batch_next_state(states=x_batch)
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1'), exist_ok=True)
    weights_before = model1._model_next_state.weights
    model1.save_checkpoint_next_state(1, 'testplayer', 'phase1')
    model1._model_next_state.load_weights(
        os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1', 'cp-000001-next-state.weights.h5'))
    weights_after = model1._model_next_state.weights
    assert weights_before == weights_after