import os
import tempfile

import numpy as np
import pytest
import torch

from app.models.behavioral_model import (
    BehavioralLSTM,
    BehavioralPredictor,
    SequenceDataset,
    train_behavioral_model,
)


class TestBehavioralLSTM:
    def test_input_output_shape(self):
        model = BehavioralLSTM(input_size=6, hidden_size=64, num_layers=2)
        batch_size = 4
        x = torch.randn(batch_size, 30, 6)
        out = model(x)
        assert out.shape == (batch_size, 1)

    def test_no_sigmoid_in_model(self):
        model = BehavioralLSTM()
        assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())

    def test_dropout_present(self):
        model = BehavioralLSTM()
        assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())

    def test_lstm_config(self):
        model = BehavioralLSTM()
        assert model.lstm.input_size == 6
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 2
        assert model.lstm.batch_first is True
        assert model.lstm.dropout == 0.3
        assert model.lstm.bidirectional is False


class TestSequenceDataset:
    def test_length_and_item(self):
        sequences = np.random.randn(10, 30, 6).astype(np.float32)
        labels = np.random.randint(0, 2, size=10).astype(np.float32)
        dataset = SequenceDataset(sequences, labels)
        assert len(dataset) == 10
        x, y = dataset[0]
        assert x.shape == (30, 6)
        assert y.shape == ()


class TestBehavioralPredictor:
    def test_predict_returns_probability(self):
        model = BehavioralLSTM(input_size=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save({"model_state_dict": model.state_dict()}, path)
            predictor = BehavioralPredictor(model_path=path, input_size=6)
            sequence = np.random.randn(30, 6).astype(np.float32)
            prob = predictor.predict(sequence)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_predict_batch_returns_probabilities(self):
        model = BehavioralLSTM(input_size=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save({"model_state_dict": model.state_dict()}, path)
            predictor = BehavioralPredictor(model_path=path, input_size=6)
            sequences = np.random.randn(5, 30, 6).astype(np.float32)
            probs = predictor.predict(sequences)
            assert probs.shape == (5,)
            assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_predictor_loads_latest_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = BehavioralLSTM(input_size=6)
            model_path = os.path.join(tmpdir, "behavioral_model_20240101_120000.pt")
            torch.save({"model_state_dict": model.state_dict()}, model_path)
            symlink_path = os.path.join(tmpdir, "behavioral_model_latest.pt")
            try:
                os.symlink(model_path, symlink_path)
            except OSError:
                import shutil
                shutil.copy2(model_path, symlink_path)
            predictor = BehavioralPredictor(model_path=symlink_path, input_size=6)
            sequence = np.random.randn(30, 6).astype(np.float32)
            prob = predictor.predict(sequence)
            assert 0.0 <= prob <= 1.0


class TestTrainBehavioralModel:
    def test_training_loop_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 50
            seq_len = 30
            n_features = 6
            sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
            labels = np.random.randint(0, 2, size=n_samples).astype(np.float32)

            seq_path = os.path.join(tmpdir, "sequences.npy")
            ids_path = os.path.join(tmpdir, "student_ids.npy")
            csv_path = os.path.join(tmpdir, "students.csv")

            np.save(seq_path, sequences)
            np.save(ids_path, np.arange(n_samples))

            import pandas as pd
            df = pd.DataFrame({
                "student_id": np.arange(n_samples),
                "at_risk": labels.astype(int),
            })
            df.to_csv(csv_path, index=False)

            models_dir = os.path.join(tmpdir, "models")
            result = train_behavioral_model(
                sequences_path=seq_path,
                student_ids_path=ids_path,
                students_csv=csv_path,
                models_dir=models_dir,
                epochs=2,
                batch_size=8,
                lr=1e-3,
            )
            assert "best_val_f1" in result
            assert result["best_epoch"] is not None
            assert os.path.exists(result["best_model_path"])

    def test_model_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 30
            seq_len = 30
            n_features = 6
            sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
            labels = np.random.randint(0, 2, size=n_samples).astype(np.float32)

            seq_path = os.path.join(tmpdir, "sequences.npy")
            ids_path = os.path.join(tmpdir, "student_ids.npy")
            csv_path = os.path.join(tmpdir, "students.csv")

            np.save(seq_path, sequences)
            np.save(ids_path, np.arange(n_samples))

            import pandas as pd
            df = pd.DataFrame({
                "student_id": np.arange(n_samples),
                "at_risk": labels.astype(int),
            })
            df.to_csv(csv_path, index=False)

            models_dir = os.path.join(tmpdir, "models")
            result = train_behavioral_model(
                sequences_path=seq_path,
                student_ids_path=ids_path,
                students_csv=csv_path,
                models_dir=models_dir,
                epochs=2,
                batch_size=8,
            )

            model_path = result["best_model_path"]
            predictor = BehavioralPredictor(model_path=model_path, input_size=6)
            prob = predictor.predict(sequences[0])
            assert 0.0 <= prob <= 1.0
