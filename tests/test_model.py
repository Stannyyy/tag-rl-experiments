import os
from pathlib import Path
import numpy as np
import pytest


class FakeTrainingLog:
    def __init__(self, loss=0.123):
        self.history = {"loss": [loss]}

class FakeModel:
    def __init__(self, num_actions=4, loss=0.123):
        self.num_actions = num_actions
        self.saved_to = None
        self.loaded_from = None
        self.history = {"loss": [loss]}

    def predict(self, x, verbose=0):
        # x is typically a numpy array with shape (batch, features...)
        # Return zeros with shape (batch, num_actions)
        if isinstance(x, list):
            # Handle the case a list is passed (unlikely with our stubs)
            batch = len(x)
        else:
            x = np.array(x)
            batch = 1 if x.ndim == 1 else (x.shape[0] if x.shape else 1)
        return np.zeros((batch, self.num_actions), dtype=float)

    def fit(self, x, y, epochs=1, verbose=0):
        # Return a fake history with a positive loss
        return FakeTrainingLog(loss=float(np.mean(np.abs(y))) if np.size(y) else 0.123)

    def save_weights(self, path):
        self.saved_to = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake-weights")

    def load_weights(self, path):
        # Simulate loading by checking file exists
        assert Path(path).exists(), f"checkpoint {path} should exist"
        self.loaded_from = path


@pytest.fixture
def patched_model(monkeypatch):
    # Patch Model.define_model to avoid importing/using the real TF backend
    import model as model_mod

    def fake_define_model(self):
        # Attach a lightweight fake model that mimics Keras API we need
        self._model = FakeModel(num_actions=getattr(self, "numActions", 4))

    monkeypatch.setattr(model_mod.Model, "define_model", fake_define_model, raising=True)
    return model_mod


def test_predict_one_and_batch_shapes(patched_model):
    # Create instance; our patched define_model will attach FakeModel
    m = patched_model.Model(experiment="/tmp-exp", learningRate=0.001, layers=[8, 8], addLSTM=False)

    # Predict one: input shape (features,) or (1, features)
    pred1 = m.predict_one(state=[0.0, 1.0, 2.0, 3.0])
    assert isinstance(pred1, np.ndarray)
    # Our FakeModel uses 4 actions by default, so len should be 4
    assert pred1.shape == (4,)

    # Predict batch: N x F -> N x A, squeezed by implementation
    X = np.random.rand(5, 4)
    predB = m.predict_batch(X)
    assert isinstance(predB, np.ndarray)
    assert predB.shape == (5, 4)


def test_train_batch_updates_losses_and_returns_summary(patched_model):
    m = patched_model.Model(experiment="/tmp-exp", learningRate=0.001, layers=[8, 8], addLSTM=False)

    # Prepare a tiny batch
    X = np.random.rand(10, 4).astype(float)
    Y = np.random.rand(10, 4).astype(float)

    before = len(m._losses)
    summary = m.train_batch(X, Y, step=42)
    after = len(m._losses)

    assert after == before + 1, "train_batch should append exactly one loss"
    assert isinstance(summary, dict)
    assert {"name", "value", "step"} <= set(summary.keys())
    assert summary["name"] == "params/losses"
    assert summary["step"] == 42
    assert isinstance(summary["value"], float)


def test_save_and_load_checkpoint(tmp_path, patched_model):
    m = patched_model.Model(experiment="/exp-test", learningRate=0.001, layers=[8, 8], addLSTM=False)

    # Build the expected checkpoint directory under cwd + experiment + ...
    cwd = Path(os.getcwd())
    # Model.save_checkpoint expects .../{experiment}/checkpoints/{name}/{phase}/cp-XXXXXX.weights.h5
    name = "unit-model"
    phase = "part1"
    checkpoint_dir = cwd / "exp-test" / "checkpoints" / name / phase
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    m.save_checkpoint(m._model, cnt=1, name=name, training_phase=phase)
    expected_file = checkpoint_dir / "cp-000001.weights.h5"
    assert expected_file.exists(), "Checkpoint file should be created by save_checkpoint"

    # Load weights back
    m.load_checkpoint(str(expected_file))
    assert getattr(m._model, "loaded_from", None) == str(expected_file), "Fake model should record the loaded path"


def test_predict_with_lstm_disabled_by_default(patched_model):
    # Ensure default addLSTM=False path is used and no reshaping errors occur
    m = patched_model.Model(experiment="/tmp-exp")
    out = m.predict_one([1, 2, 3])
    assert out.shape == (4,)


def test_multiple_calls_increase_coverage_paths(patched_model):
    # Exercise predict_batch path with float casting branch and multiple calls
    m = patched_model.Model(experiment="/tmp-exp", addLSTM=False)
    X1 = np.ones((2, 3))
    X2 = np.array([[0.1, 0.2, 0.3]], dtype=float)

    y1 = m.predict_batch(X1)
    y2 = m.predict_batch(X2)

    assert y1.shape == (2, 4)
    assert y2.shape == (1, 4)
