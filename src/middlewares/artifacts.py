import re
import yaml
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, IO, Any
from typing_extensions import TypedDict, NotRequired


from langchain.agents import AgentState
from langchain.tools import ToolRuntime
from langchain_unstructured import UnstructuredLoader
from langchain.agents.middleware.types import AgentMiddleware
from deepagents.backends.protocol import BackendFactory, BackendProtocol


logger = logging.getLogger(__name__)


class FileObject(TypedDict):
    """Type definition for file objects in state.files."""

    name: str
    content: Union[str, IO[bytes], Path]


class DocumentsState(AgentState):
    """State extension to include documents."""

    uploaded_files: NotRequired[List[FileObject]]


_SKILL_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


class ArtifactsLoaderMiddleware(AgentMiddleware):
    """
    Middleware for loading artifacts documents into the backend filesystem.

    Enforces that any file ending in SKILL.md contains valid YAML frontmatter
    compatible with SkillsMiddleware.
    """

    def __init__(
        self,
        artifacts_folder: Union[str, Path],
        backend: BackendProtocol | BackendFactory,
    ):
        super().__init__()
        self.backend = backend
        self.artifacts_base_folder_path = artifacts_folder
        self._resolved_backend: Optional[BackendProtocol] = None

        self.state_schema = DocumentsState

    def _is_base64_content(self, content: str) -> bool:
        """Detect if string content is base64 encoded.

        Args:
            content: String content to check

        Returns:
            True if content appears to be base64 encoded
        """
        # Basic heuristics for base64 detection
        if len(content) < 4:
            return False

        # Check if content has base64 characteristics
        try:
            # Base64 content should be mostly alphanumeric + / and = for padding
            base64_chars = set(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
            )
            content_chars = set(content.replace("\n", "").replace("\r", ""))

            # If more than 90% of characters are base64 chars and length is multiple of 4 (after removing whitespace)
            clean_content = content.replace("\n", "").replace("\r", "").replace(" ", "")
            if (
                len(content_chars - base64_chars) / len(content_chars) < 0.1
                and len(clean_content) % 4 == 0
                and len(clean_content) > 100
            ):  # Assume text content shorter than 100 chars is not base64

                # Try to decode to verify
                base64.b64decode(clean_content)
                return True
        except Exception:
            pass

        return False

    def _load_from_io(self, file_path: str, io_content: IO[bytes]) -> str:
        """Load document content from IO[bytes] using UnstructuredLoader.

        Args:
            file_path: File path for metadata
            io_content: IO[bytes] content to load

        Returns:
            Extracted text content
        """
        try:
            # Use UnstructuredLoader with file parameter
            loader = UnstructuredLoader(
                file=io_content,
                # Set metadata_filename to help with file type detection
                metadata_filename=file_path,
            )
            loaded_docs = loader.load()

            # Combine all content
            if loaded_docs:
                combined_content = "\n\n".join(
                    [doc.page_content for doc in loaded_docs]
                )
                return combined_content
            else:
                logger.warning(
                    f"UnstructuredLoader returned no content for {file_path}"
                )
                return ""

        except Exception as e:
            logger.error(f"Failed to load content from IO for {file_path}: {e}")
            # Try to read as text fallback
            try:
                io_content.seek(0)
                content_bytes = io_content.read()
                return content_bytes.decode("utf-8", errors="replace")
            except Exception as fallback_e:
                logger.error(
                    f"Fallback text reading also failed for {file_path}: {fallback_e}"
                )
                return ""

    def _process_state_files(
        self, state_files: List[FileObject]
    ) -> Tuple[Dict[str, str], List[Tuple[str, bytes]]]:
        """Process state files with structure {name: str, content: Union[str, IO[bytes], Path]}.

        For binary content (base64-encoded strings or IO[bytes] objects), the original
        raw bytes are preserved for upload alongside extracted text from UnstructuredLoader.

        Args:
            state_files: List of file objects with name and content fields

        Returns:
            Tuple of:
                - Dictionary mapping file paths to their extracted/raw text content
                - List of (path, bytes) tuples for raw binary files to upload
        """
        if not isinstance(state_files, list):
            raise TypeError(f"Expected list, got {type(state_files)}")

        processed_docs: Dict[str, str] = {}
        raw_binary_files: List[Tuple[str, bytes]] = []

        for i, file_obj in enumerate(state_files):
            if not isinstance(file_obj, dict):
                logger.warning(
                    f"File object at index {i} is not a dict: {type(file_obj)}"
                )
                continue

            if "name" not in file_obj:
                logger.warning(
                    f"File object at index {i} missing 'name' field: {file_obj}"
                )
                continue

            if "content" not in file_obj:
                logger.warning(
                    f"File object at index {i} missing 'content' field: {file_obj}"
                )
                continue

            name = file_obj["name"]
            content = file_obj["content"]

            # Type checking for name
            if not isinstance(name, str):
                logger.warning(f"File name at index {i} is not a string: {type(name)}")
                continue

            if not name.strip():
                logger.warning(f"File name at index {i} is empty or whitespace only")
                continue

            # Construct the file path within user_uploads folder
            file_path = (
                f"{self.artifacts_base_folder_path.rstrip('/')}/user_uploads/{name}"
            )
            extracted_path = file_path + ".extracted.txt"

            if isinstance(content, str):
                # Handle string content (could be text or base64)
                if self._is_base64_content(content):
                    # Decode base64, persist original binary, extract text
                    try:
                        decoded_bytes = base64.b64decode(content)
                        raw_binary_files.append((file_path, decoded_bytes))
                        io_content = BytesIO(decoded_bytes)
                        processed_docs[extracted_path] = self._load_from_io(
                            name, io_content
                        )
                    except Exception as e:
                        logger.error(f"Failed to decode base64 content for {name}: {e}")
                        processed_docs[file_path] = (
                            content  # Fallback to treating as text
                        )
                else:
                    # Regular text content
                    processed_docs[file_path] = content
            elif hasattr(content, "read"):
                # IO[bytes] object — persist original binary and extract text
                try:
                    content.seek(0)
                    raw_bytes = content.read()
                    raw_binary_files.append((file_path, raw_bytes))
                    io_content = BytesIO(raw_bytes)
                    processed_docs[extracted_path] = self._load_from_io(
                        name, io_content
                    )
                except Exception as e:
                    logger.error(f"Failed to read IO content for {name}: {e}")
            elif isinstance(content, (str, Path)):
                # File path - read the file
                try:
                    path_obj = Path(content)
                    if path_obj.exists():
                        with open(path_obj, "r", encoding="utf-8") as f:
                            processed_docs[file_path] = f.read()
                    else:
                        logger.warning(f"File path does not exist: {content}")
                        processed_docs[file_path] = ""
                except Exception as e:
                    logger.error(f"Failed to read file from path {content}: {e}")
                    processed_docs[file_path] = ""
            else:
                logger.warning(f"Unsupported content type for {name}: {type(content)}")
                processed_docs[file_path] = str(content)

        return processed_docs, raw_binary_files

    def wrap_tool_call(self, request, handler):
        state = request.state
        tool_runtime = request.runtime

        uploaded_files_from_state: List[FileObject] = []
        # Note: DocumentsState is a TypedDict and doesn't support isinstance() checks
        if isinstance(state, dict) and "uploaded_files" in state:
            uploaded_files_from_state = state["uploaded_files"]  # type: ignore
        if uploaded_files_from_state:
            docs_dict, raw_binary_files = self._process_state_files(
                uploaded_files_from_state
            )

            backend = self._resolved_backend
            if backend is None:
                if isinstance(self.backend, BackendProtocol):
                    backend = self.backend
                else:
                    backend = self.backend(tool_runtime)
                self._resolved_backend = backend

            # Sync the documents into the backend filesystem in artifacts/user_uploads/
            if backend:
                all_files: List[Tuple[str, bytes]] = list(raw_binary_files)
                all_files.extend(
                    (path, content.encode("utf-8"))
                    for path, content in docs_dict.items()
                )
                if all_files:
                    responses = backend.upload_files(all_files)
                    for resp in responses:
                        if resp.error:
                            logger.error(
                                f"Failed to upload artifact {resp.path}: {resp.error}"
                            )

        return handler(request)

    async def awrap_tool_call(self, request, handler):
        state = request.state
        tool_runtime = request.runtime

        uploaded_files_from_state: List[FileObject] = []
        # Note: DocumentsState is a TypedDict and doesn't support isinstance() checks
        if isinstance(state, dict) and "uploaded_files" in state:
            uploaded_files_from_state = state["uploaded_files"]  # type: ignore
        if uploaded_files_from_state:
            docs_dict, raw_binary_files = self._process_state_files(
                uploaded_files_from_state
            )

            backend = self._resolved_backend
            if backend is None:
                if isinstance(self.backend, BackendProtocol):
                    backend = self.backend
                else:
                    backend = self.backend(tool_runtime)
                self._resolved_backend = backend

            # Sync the documents into the backend filesystem in artifacts/user_uploads/
            if backend:
                all_files: List[Tuple[str, bytes]] = list(raw_binary_files)
                all_files.extend(
                    (path, content.encode("utf-8"))
                    for path, content in docs_dict.items()
                )
                if all_files:
                    responses = backend.upload_files(all_files)
                    for resp in responses:
                        if resp.error:
                            logger.error(
                                f"Failed to upload artifact {resp.path}: {resp.error}"
                            )

        return await handler(request)
