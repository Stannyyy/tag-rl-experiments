import os
import numpy as np
import pytest
import random
from model import Model, ModelNextState
from config import Config

@pytest.fixture
def models():
    config1 = Config()
    config2 = Config(); config2.add_lstm = True; config2.sequence_length_lstm = 3
    config3 = Config(); config3.layers = [10, 10, 10]
    config4 = Config(); config4.add_lstm = True; config4.sequence_length_lstm = 3; config4.layers = [10, 10, 10]
    config5 = Config(); config5.learningRate = 0.01

    return (Model(config=config1),
            Model(config=config2),
            Model(config=config3),
            Model(config=config4),
            Model(config=config5))

@pytest.fixture
def models_next_state():
    config1 = Config()
    config2 = Config(); config2.add_lstm = True; config2.sequence_length_lstm = 3
    return (ModelNextState(config=config1),
            ModelNextState(config=config2))

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
    assert pred1.shape == (model1._config.action_size,)
    assert len(model1._model.layers) == len(model1._config.layers)*2+1

    state = [list(range(model2._config.state_size))]
    experiences = [[state[0],0]] * 10
    last_x_minus_1_experiences = experiences[(model2._config.sequence_length_lstm * -1 + 1):]
    pred2 = model2.predict_one(state=[s[0] for s in last_x_minus_1_experiences] + state)
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2._config.action_size,)
    assert len(model2._model.layers) == ((len(model2._config.layers)-1)*2)+2

    pred3 = model3.predict_one(state=[[0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred3, np.ndarray)
    assert pred3.shape == (model3._config.action_size,)
    assert len(model3._model.layers) == len(model3._config.layers)*2+1

    state = [list(range(model4._config.state_size))]
    experiences = [[state[0],0]] * 10
    last_x_minus_1_experiences = experiences[(model2._config.sequence_length_lstm * -1 + 1):]
    pred4 = model4.predict_one(state=[s[0] for s in last_x_minus_1_experiences] + state)
    assert isinstance(pred4, np.ndarray)
    assert pred4.shape == (model4._config.action_size,)
    assert len(model4._model.layers) == ((len(model4._config.layers)-1)*2)+2


def test_predict_batch(models):
    model1, model2, model3, model4, _ = models

    pred1 = model1.predict_batch(states=[[0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (3, model1._config.action_size)
    assert len(model1._model.layers) == len(model1._config.layers)*2+1

    states = [[list(range(model2._config.state_size))]*3]*model2._config.batch_size
    pred2 = model2.predict_batch(states=np.array(states))
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2._config.batch_size,model2._config.action_size)
    assert len(model2._model.layers) == ((len(model2._config.layers)-1)*2)+2

    pred3 = model3.predict_batch(states=[[0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0],
                                         [0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred3, np.ndarray)
    assert pred3.shape == (3, model1._config.action_size)
    assert len(model3._model.layers) == len(model3._config.layers)*2+1

    states = [[list(range(model4._config.state_size))]*3]*model4._config.batch_size
    pred4 = model4.predict_batch(states=np.array(states))
    assert isinstance(pred4, np.ndarray)
    assert pred4.shape == (model4._config.batch_size,model4._config.action_size)
    assert len(model4._model.layers) == ((len(model4._config.layers)-1)*2)+2


def test_train_batch(models):
    model1, model2, model3, model4, model5 = models

    # Case 1
    x_batch = np.array([list(range(model1._config.state_size))]*model1._config.batch_size)
    y_batch = np.array([list(range(model1._config.action_size))]*model1._config.batch_size)

    prediction1 = model1.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    loss = model1.train_batch(x_batch=x_batch, y_batch=y_batch, log_dir='', epochs=1)
    assert loss == np.float64(21.92814)
    assert model1.losses == [21.928142547607422]

    prediction2 = model1.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == np.float64(0.013166706686314106)

    # Case 2
    x_batch = np.array([[list(range(model2._config.state_size))]*3]*model2._config.batch_size).astype(float)
    y_batch = np.array([list(range(model2._config.action_size))]*model2._config.batch_size).astype(float)

    prediction1 = model2.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    loss = model2.train_batch(x_batch=x_batch, y_batch=y_batch, log_dir='', epochs=1)
    assert loss == np.float64(22.50356)
    assert model2.losses == [22.503564834594727]

    prediction2 = model2.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == np.float64(0.005090830086481512)

    # Case 3
    x_batch = np.array([list(range(model3._config.state_size))]*model3._config.batch_size)
    y_batch = np.array([list(range(model3._config.action_size))]*model3._config.batch_size)

    prediction1 = model3.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    loss = model3.train_batch(x_batch=x_batch, y_batch=y_batch, log_dir='', epochs=1)
    assert loss == np.float64(19.39989)
    assert model3.losses == [19.399892807006836]

    prediction2 = model3.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == np.float64(0.0018020500683529354)

    # Case 4
    x_batch = np.array([[list(range(model2._config.state_size))]*3]*model2._config.batch_size).astype(float)
    y_batch = np.array([list(range(model2._config.action_size))]*model2._config.batch_size).astype(float)

    prediction1 = model4.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    loss = model4.train_batch(x_batch=x_batch, y_batch=y_batch, log_dir='', epochs=1)
    assert loss == np.float64(23.58248)
    assert model4.losses == [23.582483291625977]

    prediction2 = model4.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err-prediction2_err)/prediction1_err == np.float64(0.0006444380243421265)

    # Case 5
    x_batch = np.array([list(range(model5._config.state_size))]*model5._config.batch_size)
    y_batch = np.array([list(range(model5._config.action_size))]*model5._config.batch_size)

    prediction1 = model5.predict_batch(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch-prediction1))

    loss = model5.train_batch(x_batch=x_batch, y_batch=y_batch, log_dir='', epochs=1)
    assert loss == np.float64(21.77174)
    assert model5.losses == [21.771739959716797]

    prediction2 = model5.predict_batch(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch-prediction2))
    assert prediction1_err > prediction2_err
    assert round((prediction1_err-prediction2_err)/prediction1_err,3) == 0.009


def test_save_checkpoint(models):
    model1, _, _, _, _ = models
    x_batch = np.array([list(range(model1._config.state_size))]*model1._config.batch_size)
    model1.predict_batch(states=x_batch)
    test_path = os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1')
    os.makedirs(test_path, exist_ok=True)
    model1.save_checkpoint(os.path.join(test_path, 'cp-000001'))
    assert 'cp-000001.weights.h5' in os.listdir(test_path)


def test_load_checkpoint(models):
    model1, _, _, _, _ = models
    x_batch = np.array([list(range(model1._config.state_size))]*model1._config.batch_size)
    model1.predict_batch(states=x_batch)
    weights_before = model1._model.weights
    test_path = os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1')
    os.makedirs(test_path, exist_ok=True)
    model1.save_checkpoint(os.path.join(test_path, 'cp-000001'))
    model1._model.load_weights(os.path.join(test_path, 'cp-000001.weights.h5'))
    weights_after = model1._model.weights
    assert weights_before == weights_after


def test_define_model_predict_one_next_state(models_next_state):
    model1, model2 = models_next_state

    pred1 = model1.predict_one_next_state(state=[[0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (model1._config.state_size,)
    assert len(model1._model_next_state.layers) == len(model1._layers_next_state) * 2 + 1

    state = [list(range(model2._config.state_size))]
    experiences = [[state[0], 0]] * 10
    last_x_minus_1_experiences = experiences[(model2._config.sequence_length_lstm * -1 + 1):]
    pred2 = model2.predict_one_next_state(state=[s[0] for s in last_x_minus_1_experiences] + state)
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2._config.state_size, )
    assert len(model2._model_next_state.layers) == ((len(model2._layers_next_state) - 1) * 2) + 2


def test_predict_batch_next_state(models_next_state):
    model1, model2 = models_next_state

    pred1 = model1.predict_batch_next_state(states=[[0.0, 1.0, 2.0, 3.0],
                                                    [0.0, 1.0, 2.0, 3.0],
                                                    [0.0, 1.0, 2.0, 3.0]])
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (3, model1._config.state_size)
    assert len(model1._model_next_state.layers) == len(model1._layers_next_state) * 2 + 1

    states = [[list(range(model2._config.state_size))] * model2._config.sequence_length_lstm] * model2._config.batch_size
    pred2 = model2.predict_batch_next_state(states=np.array(states))
    assert isinstance(pred2, np.ndarray)
    assert pred2.shape == (model2._config.batch_size, model2._config.state_size)
    assert len(model2._model_next_state.layers) == ((len(model2._layers_next_state) - 1) * 2) + 2


def test_train_batch_next_state(models_next_state):
    model1, model2 = models_next_state

    # Case 1
    x_batch = np.array([list(range(model1._config.state_size))] * model1._config.batch_size)
    y_batch = np.array([list(range(model1._config.state_size))] * model1._config.batch_size)

    prediction1 = model1.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch - prediction1))

    loss = model1.train_batch_next_state(x_batch=x_batch, y_batch=y_batch)
    assert loss == np.float64(6.71816)
    assert model1.losses_next_state == [6.718158721923828]

    prediction2 = model1.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch - prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err - prediction2_err) / prediction1_err == np.float64(0.10700840562231133)

    # Case 2
    x_batch = np.array([[list(range(model2._config.state_size))] * 3] * model2._config.batch_size).astype(float)
    y_batch = np.array([list(range(model2._config.state_size))] * model2._config.batch_size).astype(float)

    prediction1 = model2.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction1.shape
    prediction1_err = np.sum(np.abs(y_batch - prediction1))

    loss = model2.train_batch_next_state(x_batch=x_batch, y_batch=y_batch)
    assert loss == np.float64(9.05004)
    assert model2.losses_next_state == [9.050043106079102]

    prediction2 = model2.predict_batch_next_state(states=x_batch)
    assert y_batch.shape == prediction2.shape
    prediction2_err = np.sum(np.abs(y_batch - prediction2))
    assert prediction1_err > prediction2_err
    assert (prediction1_err - prediction2_err) / prediction1_err == np.float64(0.06384489801845317)


def test_save_checkpoint_next_state(models_next_state):
    model1, _ = models_next_state
    x_batch = np.array([list(range(model1._config.state_size))] * model1._config.batch_size)
    model1.predict_batch_next_state(states=x_batch)
    test_path = os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1')
    os.makedirs(test_path, exist_ok=True)
    model1.save_checkpoint_next_state(os.path.join(test_path, 'cp-000001'))
    assert 'cp-000001-next-state.weights.h5' in os.listdir(test_path)


def test_load_checkpoint_next_state(models_next_state):
    model1, _ = models_next_state
    x_batch = np.array([list(range(model1._config.state_size))] * model1._config.batch_size)
    model1.predict_batch_next_state(states=x_batch)
    weights_before = model1._model_next_state.weights
    test_path = os.path.join(os.getcwd(), 'tests', 'checkpoints', 'testplayer', 'phase1')
    os.makedirs(test_path, exist_ok=True)
    model1.save_checkpoint_next_state(os.path.join(test_path, 'cp-000002'))
    model1._model_next_state.load_weights(os.path.join(test_path, 'cp-000002-next-state.weights.h5'))
    weights_after = model1._model_next_state.weights
    assert weights_before == weights_after