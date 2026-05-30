import os
from unittest.mock import patch, MagicMock

import pytest

from app.core.model_config import (
    get_text_encoder,
    get_text_model_name,
    get_model_metadata,
    get_text_model_config,
    DEFAULT_TEXT_MODEL,
    FALLBACK_TEXT_MODEL,
)


class TestGetTextModelName:
    """Tests for get_text_model_name function."""

    def test_default_returns_distilbert(self):
        """Test that default config returns DistilBERT model."""
        with patch.dict(os.environ, {}, clear=True):
            model_name = get_text_model_name()
            assert model_name == DEFAULT_TEXT_MODEL

    def test_text_model_env_var(self):
        """Test that TEXT_MODEL env var is respected."""
        with patch.dict(os.environ, {"TEXT_MODEL": "distilbert-base-uncased"}, clear=True):
            model_name = get_text_model_name()
            assert model_name == "distilbert-base-uncased"

    def test_minilm_short_name_normalized(self):
        """Test that short name 'all-MiniLM-L6-v2' is normalized to full name."""
        with patch.dict(os.environ, {"TEXT_MODEL": "all-MiniLM-L6-v2"}, clear=True):
            model_name = get_text_model_name()
            assert model_name == FALLBACK_TEXT_MODEL

    def test_fallback_flag_true(self):
        """Test that USE_MINILM_FALLBACK=true returns MiniLM."""
        with patch.dict(
            os.environ, {"USE_MINILM_FALLBACK": "true"}, clear=True
        ):
            model_name = get_text_model_name()
            assert model_name == FALLBACK_TEXT_MODEL

    def test_fallback_flag_one(self):
        """Test that USE_MINILM_FALLBACK=1 returns MiniLM."""
        with patch.dict(os.environ, {"USE_MINILM_FALLBACK": "1"}, clear=True):
            model_name = get_text_model_name()
            assert model_name == FALLBACK_TEXT_MODEL

    def test_fallback_flag_yes(self):
        """Test that USE_MINILM_FALLBACK=yes returns MiniLM."""
        with patch.dict(os.environ, {"USE_MINILM_FALLBACK": "yes"}, clear=True):
            model_name = get_text_model_name()
            assert model_name == FALLBACK_TEXT_MODEL

    def test_text_model_overrides_fallback(self):
        """Test that TEXT_MODEL takes priority over USE_MINILM_FALLBACK."""
        with patch.dict(
            os.environ,
            {"TEXT_MODEL": "distilbert-base-uncased", "USE_MINILM_FALLBACK": "true"},
            clear=True,
        ):
            model_name = get_text_model_name()
            assert model_name == "distilbert-base-uncased"


class TestGetModelMetadata:
    """Tests for get_model_metadata function."""

    def test_distilbert_metadata(self):
        """Test metadata for DistilBERT."""
        metadata = get_model_metadata("distilbert-base-uncased")
        assert metadata["hidden_size"] == 768
        assert metadata["size_mb"] == 250

    def test_minilm_metadata(self):
        """Test metadata for MiniLM."""
        metadata = get_model_metadata(FALLBACK_TEXT_MODEL)
        assert metadata["hidden_size"] == 384
        assert metadata["size_mb"] == 50

    def test_unknown_model_defaults(self):
        """Test that unknown models get default metadata."""
        metadata = get_model_metadata("unknown-model")
        assert metadata["hidden_size"] == 768
        assert metadata["size_mb"] == 250


class TestGetTextEncoder:
    """Tests for get_text_encoder function."""

    @patch("app.core.model_config.AutoTokenizer")
    @patch("app.core.model_config.AutoModel")
    def test_returns_model_and_tokenizer(self, mock_auto_model, mock_auto_tokenizer):
        """Test that get_text_encoder returns valid model and tokenizer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict(os.environ, {}, clear=True):
            model, tokenizer = get_text_encoder()

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_auto_model.from_pretrained.assert_called_once_with(DEFAULT_TEXT_MODEL)
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(DEFAULT_TEXT_MODEL)

    @patch("app.core.model_config.AutoTokenizer")
    @patch("app.core.model_config.AutoModel")
    def test_loads_minilm_when_configured(
        self, mock_auto_model, mock_auto_tokenizer
    ):
        """Test that MiniLM is loaded when configured."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict(
            os.environ, {"TEXT_MODEL": "all-MiniLM-L6-v2"}, clear=True
        ):
            model, tokenizer = get_text_encoder()

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_auto_model.from_pretrained.assert_called_once_with(FALLBACK_TEXT_MODEL)
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(FALLBACK_TEXT_MODEL)

    @patch("app.core.model_config.AutoTokenizer")
    @patch("app.core.model_config.AutoModel")
    def test_raises_on_load_failure(self, mock_auto_model, mock_auto_tokenizer):
        """Test that RuntimeError is raised when model fails to load."""
        mock_auto_model.from_pretrained.side_effect = Exception("Download failed")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                get_text_encoder()

        assert "Failed to load text encoder" in str(exc_info.value)
        assert "Download failed" in str(exc_info.value)


class TestGetTextModelConfig:
    """Tests for get_text_model_config function."""

    def test_default_config(self):
        """Test default configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_text_model_config()

        assert config["model_name"] == DEFAULT_TEXT_MODEL
        assert config["hidden_size"] == 768
        assert config["size_mb"] == 250
        assert config["is_fallback"] is False
        assert config["use_fallback_flag"] is False

    def test_fallback_config(self):
        """Test fallback configuration."""
        with patch.dict(
            os.environ, {"USE_MINILM_FALLBACK": "true"}, clear=True
        ):
            config = get_text_model_config()

        assert config["model_name"] == FALLBACK_TEXT_MODEL
        assert config["hidden_size"] == 384
        assert config["size_mb"] == 50
        assert config["is_fallback"] is True
        assert config["use_fallback_flag"] is True
