import os
import logging
from typing import Tuple, Union

from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Model configuration constants
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
FALLBACK_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Environment variable names
TEXT_MODEL_ENV_VAR = "TEXT_MODEL"
USE_MINILM_FALLBACK_ENV_VAR = "USE_MINILM_FALLBACK"

# Model metadata for reference
MODEL_METADATA = {
    "distilbert-base-uncased": {
        "hidden_size": 768,
        "size_mb": 250,
        "description": "DistilBERT base model (uncased)",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "hidden_size": 384,
        "size_mb": 50,
        "description": "MiniLM sentence transformer",
    },
    "all-MiniLM-L6-v2": {
        "hidden_size": 384,
        "size_mb": 50,
        "description": "MiniLM sentence transformer (short name)",
    },
}


def get_text_model_name() -> str:
    """
    Get the text model name from environment configuration.

    Priority:
    1. TEXT_MODEL environment variable
    2. USE_MINILM_FALLBACK=True -> all-MiniLM-L6-v2
    3. Default -> distilbert-base-uncased

    Returns:
        str: The model name to use for text encoding
    """
    # Check explicit model selection
    env_model = os.getenv(TEXT_MODEL_ENV_VAR, "").strip()
    if env_model:
        # Normalize short name to full name
        if env_model == "all-MiniLM-L6-v2":
            return FALLBACK_TEXT_MODEL
        return env_model

    # Check fallback flag
    use_fallback = os.getenv(USE_MINILM_FALLBACK_ENV_VAR, "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    if use_fallback:
        return FALLBACK_TEXT_MODEL

    return DEFAULT_TEXT_MODEL


def get_model_metadata(model_name: str) -> dict:
    """
    Get metadata for a given model name.

    Args:
        model_name: The model identifier

    Returns:
        dict: Model metadata including hidden_size and size_mb
    """
    return MODEL_METADATA.get(
        model_name,
        {
            "hidden_size": 768,
            "size_mb": 250,
            "description": "Custom model",
        },
    )


def get_text_encoder() -> Tuple[Union[AutoModel, None], Union[AutoTokenizer, None]]:
    """
    Load and return the text encoder model and tokenizer based on configuration.

    The model is selected via environment variables:
    - TEXT_MODEL: Explicit model name (e.g., 'distilbert-base-uncased', 'all-MiniLM-L6-v2')
    - USE_MINILM_FALLBACK: Set to 'true' to use MiniLM fallback

    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer

    Raises:
        RuntimeError: If the model fails to load
    """
    model_name = get_text_model_name()
    metadata = get_model_metadata(model_name)

    logger.info(
        f"Loading text encoder: {model_name} "
        f"({metadata['description']}, {metadata['hidden_size']}-dim, ~{metadata['size_mb']}MB)"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        logger.info(f"Successfully loaded text encoder: {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load text encoder '{model_name}': {e}")
        raise RuntimeError(
            f"Failed to load text encoder '{model_name}'. "
            f"Please ensure the model name is correct and you have internet connectivity "
            f"or the model is cached locally. Error: {e}"
        ) from e


def get_text_model_config() -> dict:
    """
    Get the current text model configuration.

    Returns:
        dict: Configuration including model_name, hidden_size, and fallback status
    """
    model_name = get_text_model_name()
    metadata = get_model_metadata(model_name)
    use_fallback = os.getenv(USE_MINILM_FALLBACK_ENV_VAR, "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    return {
        "model_name": model_name,
        "hidden_size": metadata["hidden_size"],
        "size_mb": metadata["size_mb"],
        "description": metadata["description"],
        "is_fallback": model_name == FALLBACK_TEXT_MODEL,
        "use_fallback_flag": use_fallback,
    }
