import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from app.preprocessing.text_processor import (
    TextEmbeddingDataset,
    cache_embeddings,
    generate_embeddings,
    load_cached_embeddings,
    load_student_labels,
    load_texts,
    process_all_texts,
)


class TestLoadTexts:
    def test_load_texts_groups_by_student(self, tmp_path):
        csv_path = tmp_path / "texts.csv"
        df = pd.DataFrame({
            "text_id": ["a", "b", "c"],
            "student_id": [1, 1, 2],
            "text": ["hello", "world", "foo"],
            "sentiment": ["positive", "positive", "negative"],
            "word_count": [1, 1, 1],
        })
        df.to_csv(csv_path, index=False)
        result = load_texts(str(csv_path))
        assert result[1] == "hello world"
        assert result[2] == "foo"


class TestLoadStudentLabels:
    def test_load_labels(self, tmp_path):
        csv_path = tmp_path / "students.csv"
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "at_risk": [0, 1, 0],
        })
        df.to_csv(csv_path, index=False)
        result = load_student_labels(str(csv_path))
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == 0


class TestTextEmbeddingDataset:
    def test_dataset_length(self):
        texts = {1: "hello", 2: "world"}
        labels = {1: 0, 2: 1}
        dataset = TextEmbeddingDataset(texts, labels)
        assert len(dataset) == 2

    def test_dataset_getitem(self):
        texts = {1: "hello", 2: "world"}
        labels = {1: 0, 2: 1}
        dataset = TextEmbeddingDataset(texts, labels)
        sid, text, label = dataset[0]
        assert sid == 1
        assert text == "hello"
        assert label == 0

    def test_dataset_without_labels(self):
        texts = {1: "hello"}
        dataset = TextEmbeddingDataset(texts)
        sid, text, label = dataset[0]
        assert sid == 1
        assert text == "hello"
        assert label is None


class MockModel:
    def __init__(self, batch_size, hidden_size=768):
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def __call__(self, **kwargs):
        seq_len = 10
        return type("Outputs", (), {
            "last_hidden_state": torch.randn(self.batch_size, seq_len, self.hidden_size)
        })()

    def to(self, device):
        return self

    def eval(self):
        pass

    def parameters(self):
        return []


class MockTokenizer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        seq_len = 10
        return {
            "input_ids": torch.randint(0, 100, (self.batch_size, seq_len)),
            "attention_mask": torch.ones(self.batch_size, seq_len),
        }


class TestGenerateEmbeddings:
    def test_generate_embeddings_shape(self):
        texts = ["hello world", "foo bar"]
        model = MockModel(batch_size=2)
        tokenizer = MockTokenizer(batch_size=2)
        embeddings = generate_embeddings(texts, model, tokenizer, batch_size=2)

        assert embeddings.shape == (2, 768)

    def test_generate_embeddings_multiple_batches(self):
        texts = ["a", "b", "c"]
        embeddings_list = []

        class TrackingModel:
            def __init__(self):
                self.call_count = 0

            def __call__(self, **kwargs):
                self.call_count += 1
                return type("Outputs", (), {
                    "last_hidden_state": torch.randn(1, 10, 768)
                })()

            def to(self, device):
                return self

            def eval(self):
                pass

        model = TrackingModel()
        tokenizer = MockTokenizer(batch_size=1)
        embeddings = generate_embeddings(texts, model, tokenizer, batch_size=1)

        assert embeddings.shape == (3, 768)
        assert model.call_count == 3


class TestCacheAndLoad:
    def test_cache_and_load_roundtrip(self, tmp_path):
        embeddings = np.random.randn(5, 768).astype(np.float32)
        student_ids = np.array([1, 2, 3, 4, 5])

        cache_path = str(tmp_path / "embeddings.npy")
        ids_path = str(tmp_path / "ids.npy")

        cache_embeddings(embeddings, student_ids, cache_path, ids_path)
        loaded = load_cached_embeddings(cache_path, ids_path)

        assert loaded is not None
        loaded_embeddings, loaded_ids = loaded
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        np.testing.assert_array_equal(student_ids, loaded_ids)

    def test_load_cached_missing_returns_none(self, tmp_path):
        cache_path = str(tmp_path / "nonexistent.npy")
        ids_path = str(tmp_path / "nonexistent_ids.npy")
        result = load_cached_embeddings(cache_path, ids_path)
        assert result is None


class TestProcessAllTexts:
    @patch("app.preprocessing.text_processor.get_text_encoder")
    @patch("app.preprocessing.text_processor.load_cached_embeddings")
    def test_uses_cache_when_available(self, mock_load_cached, mock_get_encoder, tmp_path):
        mock_embeddings = np.random.randn(3, 768).astype(np.float32)
        mock_ids = np.array([1, 2, 3])
        mock_load_cached.return_value = (mock_embeddings, mock_ids)

        texts_csv = str(tmp_path / "texts.csv")
        students_csv = str(tmp_path / "students.csv")

        embeddings, student_ids = process_all_texts(
            texts_csv=texts_csv,
            students_csv=students_csv,
            cache_path=str(tmp_path / "cache.npy"),
            ids_path=str(tmp_path / "ids.npy"),
        )

        np.testing.assert_array_equal(embeddings, mock_embeddings)
        np.testing.assert_array_equal(student_ids, mock_ids)
        mock_get_encoder.assert_not_called()

    @patch("app.preprocessing.text_processor.get_text_encoder")
    @patch("app.preprocessing.text_processor.load_cached_embeddings")
    def test_recomputes_when_forced(self, mock_load_cached, mock_get_encoder, tmp_path):
        mock_load_cached.return_value = (np.array([1]), np.array([1]))

        texts_csv = tmp_path / "texts.csv"
        students_csv = tmp_path / "students.csv"
        pd.DataFrame({
            "text_id": ["a"],
            "student_id": [1],
            "text": ["hello"],
            "sentiment": ["positive"],
            "word_count": [1],
        }).to_csv(texts_csv, index=False)
        pd.DataFrame({
            "student_id": [1],
            "at_risk": [0],
        }).to_csv(students_csv, index=False)

        mock_get_encoder.return_value = (MockModel(batch_size=1), MockTokenizer(batch_size=1))

        embeddings, student_ids = process_all_texts(
            texts_csv=str(texts_csv),
            students_csv=str(students_csv),
            cache_path=str(tmp_path / "cache.npy"),
            ids_path=str(tmp_path / "ids.npy"),
            force_recompute=True,
        )

        assert embeddings.shape == (1, 768)
        assert student_ids.shape == (1,)

    @patch("app.preprocessing.text_processor.get_text_encoder")
    @patch("app.preprocessing.text_processor.load_cached_embeddings")
    def test_output_shape(self, mock_load_cached, mock_get_encoder, tmp_path):
        mock_load_cached.return_value = None

        texts_csv = tmp_path / "texts.csv"
        students_csv = tmp_path / "students.csv"
        pd.DataFrame({
            "text_id": ["a", "b"],
            "student_id": [1, 2],
            "text": ["hello", "world"],
            "sentiment": ["positive", "negative"],
            "word_count": [1, 1],
        }).to_csv(texts_csv, index=False)
        pd.DataFrame({
            "student_id": [1, 2],
            "at_risk": [0, 1],
        }).to_csv(students_csv, index=False)

        mock_get_encoder.return_value = (MockModel(batch_size=2), MockTokenizer(batch_size=2))

        embeddings, student_ids = process_all_texts(
            texts_csv=str(texts_csv),
            students_csv=str(students_csv),
            cache_path=str(tmp_path / "cache.npy"),
            ids_path=str(tmp_path / "ids.npy"),
        )

        assert embeddings.shape[1] == 768
        assert len(student_ids) == embeddings.shape[0]
