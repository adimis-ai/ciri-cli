# CIRI Copilot

A command-line interface for managing and interacting with the Ciri AI agent system. The Ciri Copilot CLI provides tools for compiling agents, streaming conversations, managing conversation history, and more.

## Features

- **Agent Compilation**: Compile Ciri agents with SQLite persistence and in memory caching
- **Streaming Interface**: Stream responses from the agent in real-time
- **Synchronous Invocation**: Get single responses from the agent
- **Conversation History**: Retrieve and filter conversation history by thread
- **Flexible Configuration**: Support for JSON and YAML configuration files
- **Database Integration**: Built-in SQLite support for state persistence and cross-thread memory
- **In Memory Caching**: Optional LLM response caching for improved performance
- **In-Memory Caching**: LLM response caching is handled in-process via an InMemoryCache

## Installation

### Prerequisites

- Python 3.10 or higher

### Setup

1. Clone the repository and navigate to the src-copilot directory:
```bash
cd src-copilot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Required environment variables:
```bash
SQLITE_URL=
```

## Usage

The CLI provides four main commands: `compile`, `stream`, `invoke`, and `history`.

### General Options

```bash
ciri [OPTIONS] COMMAND [ARGS]

Options:
  -v, --verbose          Enable verbose/debug logging
  --sqlite-url TEXT    SQLite URL for storage (overrides SQLITE_URL env var)
  -h, --help            Show help message and exit
```

## Commands

### 1. Compile

Compile the Ciri agent with database configuration. In-memory caching is used automatically.

```bash
ciri compile --spec <SPEC> [OPTIONS]
```

#### Arguments

- `--spec TEXT` (required): Ciri specification as JSON string or path to JSON/YAML file

#### Options

- `--debug`: Enable debug mode for the agent

#### Examples

**Using a JSON file:**
```bash
ciri compile --spec ./config/ciri-spec.json
```

**Using a YAML file:**
```bash
ciri compile --spec ./config/ciri-spec.yaml --debug
```

**Using inline JSON:**
```bash
ciri compile --spec '{"model": "gpt-4", "temperature": 0.7}'
```

#### Ciri Spec Format

**JSON format (ciri-spec.json):**
```json
{
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 2000,
  "system_prompt": "You are Ciri, a helpful AI assistant."
}
```

**YAML format (ciri-spec.yaml):**
```yaml
model: gpt-4
temperature: 0.7
max_tokens: 2000
system_prompt: "You are Ciri, a helpful AI assistant."
```

---

### 2. Stream

Stream responses from the Ciri agent in real-time.

```bash
ciri stream --input <INPUT> [OPTIONS]
```

#### Arguments

- `--input TEXT` (required): Input message state or command (JSON string)

#### Options

- `--thread-id TEXT`: Thread ID for conversation persistence
- `--config TEXT`: Runnable configuration (JSON string)
- `--no-subgraphs`: Disable subgraph streaming
- `--stream-mode TEXT`: Stream mode (e.g., 'values', 'updates', 'debug')

#### Examples

**Basic streaming:**
```bash
ciri stream --input '{"messages": [{"role": "user", "content": "Hello, Ciri!"}]}'
```

**With thread persistence:**
```bash
ciri stream \
  --input '{"messages": [{"role": "user", "content": "What is AI?"}]}' \
  --thread-id conversation-123
```

**With custom stream mode:**
```bash
ciri stream \
  --input '{"messages": [{"role": "user", "content": "Explain quantum computing"}]}' \
  --thread-id tech-thread \
  --stream-mode updates
```

**Continue a conversation:**
```bash
# First message
ciri stream \
  --input '{"messages": [{"role": "user", "content": "Tell me about Python"}]}' \
  --thread-id python-tutorial

# Follow-up in the same thread
ciri stream \
  --input '{"messages": [{"role": "user", "content": "What about list comprehensions?"}]}' \
  --thread-id python-tutorial
```

---

### 3. Invoke

Invoke the Ciri agent and get a single response (non-streaming).

```bash
ciri invoke --input <INPUT> [OPTIONS]
```

#### Arguments

- `--input TEXT` (required): Input message state or command (JSON string)

#### Options

- `--thread-id TEXT`: Thread ID for conversation persistence
- `--config TEXT`: Runnable configuration (JSON string)

#### Examples

**Basic invocation:**
```bash
ciri invoke --input '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

**With thread ID:**
```bash
ciri invoke \
  --input '{"messages": [{"role": "user", "content": "Explain machine learning"}]}' \
  --thread-id ml-session
```

**With custom configuration:**
```bash
ciri invoke \
  --input '{"messages": [{"role": "user", "content": "Hello"}]}' \
  --config '{"configurable": {"model": "gpt-4", "temperature": 0.5}}'
```

---

### 4. History

Retrieve conversation history for a specific thread.

```bash
ciri history --thread-id <THREAD_ID> [OPTIONS]
```

#### Arguments

- `--thread-id TEXT` (required): Thread ID to retrieve history for

#### Options

- `--filter TEXT`: Filter criteria (JSON string)
- `--before TEXT`: Get history before this config (JSON string)
- `--limit INTEGER`: Maximum number of history entries to retrieve

#### Examples

**Get all history for a thread:**
```bash
ciri history --thread-id conversation-123
```

**Limit number of entries:**
```bash
ciri history --thread-id conversation-123 --limit 10
```

**With filter:**
```bash
ciri history \
  --thread-id conversation-123 \
  --filter '{"checkpoint_ns": "main"}' \
  --limit 20
```

**Get history before a specific checkpoint:**
```bash
ciri history \
  --thread-id conversation-123 \
  --before '{"configurable": {"checkpoint_id": "abc123"}}'
```

## Architecture

### Components

#### CiriConfig

A configuration dataclass that centralizes all controller settings:

- **Configuration Management**: Centralized settings for SQLite and embeddings
- **Environment Integration**: Automatic loading from environment variables
- **Validation**: Built-in configuration validation
- **Backward Compatibility**: Supports legacy constructor parameters

```python
from src.controllers import CiriConfig, CiriController

# Create config from environment variables
config = CiriConfig.from_env()

# Or create custom config
config = CiriConfig(
  sqlite_url="",
  embedding_dims=1024
)

# Use with controller
controller = CiriController(config=config)
```

#### CiriController

The main controller class that manages the Ciri agent lifecycle:

- **SqliteStore**: Cross-thread memory and storage with vector embeddings
- **SqliteSaver**: Conversation persistence and checkpointing
 - **In-Memory Cache**: In-process caching via `InMemoryCache` for improved performance
- **CompiledStateGraph**: The compiled agent graph
- **Configuration-Driven**: Uses CiriConfig for centralized settings
- **Resource Management**: Automatic cleanup via context manager

#### _CacheManager

Internal singleton manager for expensive resources:

- **Embeddings Cache**: Reuses embedding models across instances
- **Store Cache**: Manages SqliteStore instances per database URL
- **Checkpointer Cache**: Manages SqliteSaver instances per database URL
- **Lifecycle Management**: Proper initialization and cleanup

#### Agent Lifecycle

1. **Create Configuration**: Use `CiriConfig.from_env()` or create custom config
2. **Initialize Controller**: Create a `CiriController` instance with configuration
3. **Compile Agent**: Load Ciri spec and compile with `controller.compile()`
4. **Interact**: Use `stream()` or `invoke()` to interact with the agent
5. **Retrieve History**: Use `history()` to access past conversations
6. **Cleanup**: Controller automatically cleans up resources via context manager or `close()`

### Database Integration

The controller now supports improved database integration:

- **Vector Embeddings**: Configurable embedding dimensions and indexing fields
- **Cross-Thread Memory**: Shared state across conversation threads
- **Efficient Caching**: Singleton management of expensive resources
- **Connection Pooling**: Automatic connection management

### Database Schema

The CLI automatically sets up required database tables:

- **Checkpoints**: Stores conversation state snapshots
- **Store**: Cross-thread memory and shared data
- **Writes**: Pending state modifications

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `SQLITE_URL` | Yes | SQLite/SQLite connection URL | None |
| `EMBEDDING_DIMS` | No | Embedding vector dimensions | 1024 |
| `EMBEDDING_INDEX_FIELDS` | No | Fields to index for embeddings | "$" |

### Example .env file

```bash
# Database Configuration
SQLITE_URL=

# Optional: Embedding Configuration
EMBEDDING_DIMS=1024
EMBEDDING_INDEX_FIELDS="$"

# Optional: Logging
LOG_LEVEL=INFO
```

## Advanced Usage

### Programmatic API Usage

The refactored `CiriController` can be used programmatically with the new configuration system:

```python
from src.controllers import CiriController, CiriConfig
from src.agent import Ciri

# Method 1: Use environment variables
controller = CiriController()

# Method 2: Custom configuration
config = CiriConfig(
  sqlite_url="",
  embedding_dims=512,
  embedding_index_fields="content,metadata"
)
controller = CiriController(config=config)

# Use as context manager for automatic cleanup
with controller:
  ciri = Ciri.from_spec(spec_dict)
  controller.compile(ciri, debug=True)
    
  # Stream responses
  for item in controller.stream(input_data, thread_id="session-1"):
    if item["type"] == "messages":
      print(item["data"]["token"], end="")
    elif item["type"] == "interrupt":
      print(f"\nInterrupt: {item['data']}")
```

### Configuration Validation

The new `CiriConfig` class provides built-in validation:

```python
try:
    config = CiriConfig.from_env()
    config.validate()  # Automatically called in CiriController.__init__
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Custom Configuration Files

Create reusable configuration files for different agent profiles:

**config/research-agent.yaml:**
```yaml
model: gpt-4
temperature: 0.3
max_tokens: 3000
system_prompt: |
  You are a research assistant specializing in academic research.
  Provide detailed, well-sourced answers with citations.
```

**config/creative-agent.yaml:**
```yaml
model: gpt-4
temperature: 0.9
max_tokens: 2000
system_prompt: |
  You are a creative writing assistant.
  Help users with storytelling, poetry, and creative content.
```

Usage:
```bash
ciri compile --spec config/research-agent.yaml
ciri compile --spec config/creative-agent.yaml
```

### Chaining Commands

Compile and immediately start a conversation:
```bash
ciri compile --spec config/agent.yaml && \
ciri stream --input '{"messages": [{"role": "user", "content": "Hello!"}]}' --thread-id session-1
```

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
ciri --verbose compile --spec config/agent.yaml --debug
```

## Development

### Project Structure

```
src-copilot/
├── src/
│   ├── __main__.py       # CLI entry point
│   ├── controllers.py    # CiriController implementation
│   ├── agent.py          # Ciri agent definition
│   └── ...
├── config/               # Configuration files
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines:
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

**1. Configuration Error**
```
ValueError: SQLite URL is required. Set SQLITE_URL environment variable or provide sqlite_url.
```
Solution: Set the `SQLITE_URL` environment variable, use `--sqlite-url` flag, or provide a `CiriConfig` with `sqlite_url`.

**2. Agent Not Compiled**
```
ValueError: CiriController is not compiled. Call compile() before using this method.
```
Solution: Run `ciri compile --spec <spec>` before using `stream`, `invoke`, or `history` commands.

**3. Invalid JSON Input**
```
Error: Invalid JSON: Expecting value: line 1 column 1 (char 0)
```
Solution: Ensure your JSON strings are properly formatted and quoted.

**5. Embedding Configuration Issues**
```
Error: Invalid embedding dimensions or index fields
```
Solution: Check `EMBEDDING_DIMS` is a valid integer and `EMBEDDING_INDEX_FIELDS` contains valid field names.

### Getting Help

```bash
# General help
ciri --help

# Command-specific help
ciri compile --help
ciri stream --help
ciri invoke --help
ciri history --help
```

## License

[Your License Here]

## Contributing

[Contributing guidelines here]

## Support

For issues and questions:
- GitHub Issues: [Your repo URL]
- Documentation: [Your docs URL]
- Email: [Your email]