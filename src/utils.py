import os
import asyncio
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def initialize_embeddings():
    """
    Initialize Hugging Face embeddings using configuration from Django settings.
    """
    embedding_model = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    normalize = os.getenv("EMBEDDING_NORMALIZE", "True") == "True"
    model_kwargs = os.getenv("EMBEDDING_MODEL_KWARGS", {})
    encode_kwargs = os.getenv("EMBEDDING_ENCODE_KWARGS", {})

    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": device,
            **model_kwargs,
        },
        encode_kwargs={
            "normalize_embeddings": normalize,
            **encode_kwargs,
        },
    )


def install_embedding_model():
    """
    Pre-install/download the default embedding model on startup.
    This prevents API calls from being blocked by initial download.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Pre-installing embedding model")
    try:
        initialize_embeddings()
        logger.info("Successfully installed embedding model")
    except Exception as e:
        logger.error(f"Failed to install embedding model: {e}")
