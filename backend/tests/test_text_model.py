import os
import tempfile

import numpy as np
import pytest
import torch

from app.models.text_model import (
    EmbeddingDataset,
    TextClassifier,
    TextPredictor,
    train_text_model,
)


class TestTextClassifier:
    def test_input_output_shape(self):
        model = TextClassifier(input_dim=768, hidden_dim=128)
        batch_size = 4
        x = torch.randn(batch_size, 768)
        out = model(x)
        assert out.shape == (batch_size, 1)

    def test_no_sigmoid_in_model(self):
        model = TextClassifier()
        assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())

    def test_dropout_present(self):
        model = TextClassifier()
        assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())


class TestEmbeddingDataset:
    def test_length_and_item(self):
        embeddings = np.random.randn(10, 768).astype(np.float32)
        labels = np.random.randint(0, 2, size=10).astype(np.float32)
        dataset = EmbeddingDataset(embeddings, labels)
        assert len(dataset) == 10
        x, y = dataset[0]
        assert x.shape == (768,)
        assert y.shape == ()


class TestTextPredictor:
    def test_predict_returns_probability(self):
        model = TextClassifier(input_dim=768)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save({"model_state_dict": model.state_dict()}, path)
            predictor = TextPredictor(model_path=path, input_dim=768)
            embedding = np.random.randn(768).astype(np.float32)
            prob = predictor.predict(embedding)
            assert 0.0 <= prob <= 1.0

    def test_predict_batch_returns_probabilities(self):
        model = TextClassifier(input_dim=768)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save({"model_state_dict": model.state_dict()}, path)
            predictor = TextPredictor(model_path=path, input_dim=768)
            embeddings = np.random.randn(5, 768).astype(np.float32)
            probs = predictor.predict_batch(embeddings)
            assert probs.shape == (5,)
            assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_predictor_loads_latest_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TextClassifier(input_dim=768)
            model_path = os.path.join(tmpdir, "text_model_20240101_120000.pt")
            torch.save({"model_state_dict": model.state_dict()}, model_path)
            symlink_path = os.path.join(tmpdir, "text_model_latest.pt")
            try:
                os.symlink(model_path, symlink_path)
            except OSError:
                import shutil
                shutil.copy2(model_path, symlink_path)
            predictor = TextPredictor(model_path=symlink_path, input_dim=768)
            embedding = np.random.randn(768).astype(np.float32)
            prob = predictor.predict(embedding)
            assert 0.0 <= prob <= 1.0


class TestTrainTextModel:
    def test_training_loop_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 50
            embeddings = np.random.randn(n_samples, 32).astype(np.float32)
            labels = np.random.randint(0, 2, size=n_samples).astype(np.float32)

            emb_path = os.path.join(tmpdir, "embeddings.npy")
            ids_path = os.path.join(tmpdir, "student_ids.npy")
            csv_path = os.path.join(tmpdir, "students.csv")

            np.save(emb_path, embeddings)
            np.save(ids_path, np.arange(n_samples))

            import pandas as pd
            df = pd.DataFrame({
                "student_id": np.arange(n_samples),
                "at_risk": labels.astype(int),
            })
            df.to_csv(csv_path, index=False)

            models_dir = os.path.join(tmpdir, "models")
            result = train_text_model(
                embeddings_path=emb_path,
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
            embeddings = np.random.randn(n_samples, 16).astype(np.float32)
            labels = np.random.randint(0, 2, size=n_samples).astype(np.float32)

            emb_path = os.path.join(tmpdir, "embeddings.npy")
            ids_path = os.path.join(tmpdir, "student_ids.npy")
            csv_path = os.path.join(tmpdir, "students.csv")

            np.save(emb_path, embeddings)
            np.save(ids_path, np.arange(n_samples))

            import pandas as pd
            df = pd.DataFrame({
                "student_id": np.arange(n_samples),
                "at_risk": labels.astype(int),
            })
            df.to_csv(csv_path, index=False)

            models_dir = os.path.join(tmpdir, "models")
            result = train_text_model(
                embeddings_path=emb_path,
                student_ids_path=ids_path,
                students_csv=csv_path,
                models_dir=models_dir,
                epochs=2,
                batch_size=8,
            )

            model_path = result["best_model_path"]
            predictor = TextPredictor(model_path=model_path, input_dim=16)
            prob = predictor.predict(embeddings[0])
            assert 0.0 <= prob <= 1.0
