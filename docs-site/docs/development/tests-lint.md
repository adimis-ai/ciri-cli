# Tests & Linting

Testing
- Run unit tests with pytest:

  pytest

- Async tests: pytest-asyncio is available
- Coverage: pytest --cov=src

Linting & Formatting
- Ruff (fast linter):

  ruff check src/
  ruff check src/ --fix

- Black (formatter):

  black src/

- Mypy (type checking):

  mypy src/

Pre-PR checklist (recommended):
- ruff check --fix
- black --check src/
- pytest

CI pipelines should run the same commands to ensure consistency.

---

## Typical CI job flow

```mermaid
sequenceDiagram
  participant Dev
  participant Repo
  participant CI
  participant Runner
  Dev->>Repo: push PR
  Repo->>CI: trigger workflow
  CI->>Runner: install deps
  Runner->>Runner: run ruff, black, mypy
  Runner->>Runner: run pytest
  Runner-->>CI: report status
  CI-->>Repo: show checks
```

