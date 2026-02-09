import os
import asyncio
import platform
from pathlib import Path
from dotenv import load_dotenv, set_key


def get_app_data_dir() -> Path:
    """Determine OS-specific application data directory for Ciri."""
    system = platform.system()
    user_home = Path.home()

    if system == "Windows":
        root = user_home / "AppData" / "Local" / "Ciri"
    elif system == "Darwin":
        root = user_home / "Library" / "Application Support" / "Ciri"
    else:  # Linux and others
        root = user_home / ".local" / "share" / "ciri"

    root.mkdir(parents=True, exist_ok=True)
    return root


def load_all_dotenv():
    """Load .env from current directory and global app data directory."""
    # Load from current directory
    load_dotenv()
    # Load from global app data directory
    global_env = get_app_data_dir() / ".env"
    if global_env.exists():
        load_dotenv(dotenv_path=global_env, override=False)


# def initialize_embeddings():
#     """
#     Initialize Hugging Face embeddings using configuration from Django settings.
#     """
#     embedding_model = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
#     device = os.getenv("EMBEDDING_DEVICE", "cpu")
#     normalize = os.getenv("EMBEDDING_NORMALIZE", "True") == "True"
#     model_kwargs = os.getenv("EMBEDDING_MODEL_KWARGS", {})
#     encode_kwargs = os.getenv("EMBEDDING_ENCODE_KWARGS", {})

#     return HuggingFaceEmbeddings(
#         model_name=embedding_model,
#         model_kwargs={
#             "device": device,
#             **model_kwargs,
#         },
#         encode_kwargs={
#             "normalize_embeddings": normalize,
#             **encode_kwargs,
#         },
#     )


# def install_embedding_model():
#     """
#     Pre-install/download the default embedding model on startup.
#     This prevents API calls from being blocked by initial download.
#     """
#     import logging

#     logger = logging.getLogger(__name__)

#     logger.info("Pre-installing embedding model")
#     try:
#         initialize_embeddings()
#         logger.info("Successfully installed embedding model")
#     except Exception as e:
#         logger.error(f"Failed to install embedding model: {e}")


def get_default_filesystem_root() -> Path:
    """Return the current working directory as the default filesystem root."""
    return Path(os.getcwd()).resolve()

