# Toolkits Guide (MCP)

Toolkits are **MCP (Model Context Protocol) servers** that give Ciri live, bidirectional access to external services — databases, SaaS APIs, cloud platforms, internal tools, and anything else that can be wrapped in an MCP interface. Ciri's `ToolkitInjectionMiddleware` acts as an MCP client that discovers, connects to, and hot-reloads these servers without requiring a restart.

---

## What Is a Toolkit?

Where a Skill is a playbook (static knowledge), a Toolkit is a **live service connection**. When Ciri has a Slack toolkit, she can actually send messages, read channels, and search conversations — not just know how to do it.

Toolkits are standard MCP servers. Anything that speaks MCP can be a Ciri toolkit:
- **Community MCP servers** (Slack, GitHub, Google Drive, Postgres, Jira, etc.)
- **Custom servers** built by Ciri using the `mcp-builder` skill
- **Internal systems** exposed via a lightweight FastMCP or TypeScript MCP SDK wrapper

---

## Architecture

```mermaid
flowchart LR
  subgraph Ciri
    MW[ToolkitInjectionMiddleware\n MCP Client]
  end

  subgraph Core Harness
    T1[~/.local/share/ciri/toolkits/slack/]
    T2[~/.local/share/ciri/toolkits/github/]
  end

  subgraph Project Harness
    T3[.ciri/toolkits/my-db/]
  end

  MW <--> T1
  MW <--> T2
  MW <--> T3
  T1 <--> Slack[Slack API]
  T2 <--> GitHub[GitHub API]
  T3 <--> DB[(PostgreSQL)]
```

The `ToolkitInjectionMiddleware`:
1. Discovers toolkit directories in the core and project harness
2. Spawns each MCP server as a subprocess (Python or Node.js)
3. Registers all server-exposed tools into Ciri's active toolset
4. Monitors for version changes and restarts servers automatically

---

## Adding a Toolkit

### Via Ciri (recommended)

Ask Ciri in natural language:

```
You > Build a toolkit to connect to my PostgreSQL database at localhost:5432
You > Create a Slack toolkit for our workspace
You > Build a toolkit for our internal Jira instance at jira.company.com
You > Connect to our GitHub organization repos via toolkit
```

Ciri invokes the `toolkit_builder_agent`, which uses the `mcp-builder` skill to:
1. Scaffold the MCP server (Python with FastMCP, or TypeScript with MCP SDK)
2. Write the tool implementations
3. Store the server in `.ciri/toolkits/<name>/` or the core harness
4. Test the connection
5. Hot-reload without restart

### Via `/sync`

```
You > /sync — add toolkits for the services in my stack
```

The Trainer Agent detects service configurations (database URLs, API keys, etc.) from your workspace and builds appropriate toolkits.

### Manual addition (community MCP servers)

For well-known community MCP servers, add the configuration to `.ciri/settings.json`:

```json
{
  "toolkits": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/Documents"],
      "transport": "stdio"
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost/mydb"
      },
      "transport": "stdio"
    }
  }
}
```

---

## Toolkit Directory Structure

A Ciri-built toolkit lives in a directory with either `pyproject.toml` (Python) or `package.json` (Node.js):

**Python toolkit (FastMCP):**
```
.ciri/toolkits/my-toolkit/
├── pyproject.toml     ← Package metadata + dependencies
├── server.py          ← MCP server implementation
└── .env               ← Secrets (never committed)
```

**Node.js toolkit:**
```
.ciri/toolkits/my-toolkit/
├── package.json       ← Package metadata + dependencies
├── index.js           ← MCP server implementation
└── .env               ← Secrets (never committed)
```

The `ToolkitInjectionMiddleware` auto-installs dependencies via `uv pip install` (Python) or `npm install` (Node.js) on first run.

---

## Toolkit Discovery Order

Toolkits are discovered using an ordered dictionary — **first-wins**:

1. Core harness (`~/.local/share/ciri/toolkits/`) — highest priority
2. Project harness (`.ciri/toolkits/`) — project-specific

If a toolkit with the same name exists in both harness levels, the **core harness version is used**.

---

## Security Practices

**Process isolation**: Each MCP server runs as a separate subprocess. A crash or misbehavior in a toolkit does not affect Ciri or other toolkits.

**Secrets management**: Store API keys and credentials in the toolkit's `.env` file. Ciri will never hardcode credentials into toolkit source code. Add `.ciri/toolkits/*/` to `.gitignore` if your toolkit directories contain secrets.

**Scoping**: Each toolkit only has access to the services it explicitly implements. A Slack toolkit cannot access your database.

**Transport**: CIRI uses `stdio` transport by default (subprocess stdin/stdout). SSE transport (HTTP) is supported for remote MCP servers.

---

## Toolkit vs. Skill vs. SubAgent

| | Toolkit | Skill | SubAgent |
|---|---|---|---|
| **What it is** | Live MCP server process | Static playbook + scripts | Specialized agent instance |
| **When to use** | External API/service calls | Domain knowledge + workflows | Different reasoning scope |
| **Created by** | `toolkit_builder_agent` | `skill_builder_agent` | `subagent_builder_agent` |
| **Storage** | `toolkits/` directory | `skills/` directory | `subagents/` directory |
| **Hot-reload** | Yes (on `/sync`) | Yes (on `/sync`) | Yes (on `/sync`) |

---

## Hot-Reload

Toolkits are hot-reloaded when:
- `/sync` is run
- A toolkit's `pyproject.toml` or `package.json` version changes
- Ciri restarts

No manual restart required after adding or updating a toolkit.
