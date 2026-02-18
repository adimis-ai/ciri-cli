# Examples & Tutorials

Hands-on walkthroughs showing Ciri in action across different domains and workflows.

---

## First Session

The minimal path from install to first useful output.

```bash
# 1. Install
pip install ciri

# 2. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Start
ciri
```

```
 ██████╗██╗██████╗ ██╗
██╔════╝██║██╔══██╗██║
██║     ██║██████╔╝██║
██║     ██║██╔══██╗██║
╚██████╗██║██║  ██║██║

Ciri  —  Your local AI copilot

✓ Playwright ready
✓ API configured (claude-sonnet-4-6)
✓ Database ready
✓ Ready

You > /new-thread
```

Then run `/sync` to let Ciri learn your workspace:

```
You > /sync
Ciri > Running workspace analysis...
       ✓ Scanned project structure
       ✓ Identified domain: SaaS B2B application
       ✓ Wrote .ciri/memory/AGENT.md
       ✓ Wrote .ciri/memory/architecture.md
       ✓ Registered 3 new skills from core harness
       Workspace sync complete.
```

---

## Software Engineering

### Review and Refactor a Module

```
You > Review @files:src/auth/service.py and suggest improvements
```

Ciri reads the file, analyzes it against your workspace conventions (from `.ciri/memory/`), and proposes specific changes with explanations.

### Write and Run a Migration Script

```
You > Write a Python script to backfill the `created_at` column in the `users` table
      for all rows where it is NULL, using the value from `registration_date`
```

Ciri writes the script. Before executing it:

```
╭─────────────────────────────────────────────╮
│  ⚙️  execute_script                           │
│  Language: python                             │
│  Script: backfill_created_at.py              │
│  Dependencies: psycopg2-binary, python-dotenv│
│                                              │
│  [a] Approve  [e] Edit  [r] Reject           │
╰─────────────────────────────────────────────╯
```

Inspect and approve. Ciri runs it in an isolated venv, streams output, and saves results.

### Explain an Unfamiliar Codebase

```
You > I just joined this project. Walk me through the data flow from a user submitting
      a form on the frontend to the database write on the backend
```

Ciri uses the memory files (written by `/sync`) to give a contextual walkthrough, pointing to exact files and functions.

---

## Marketing & Content

### Draft a Q1 Investor Update

```
You > Draft the Q1 investor update based on the metrics in @files:reports/q1-metrics.xlsx
      and the product milestones in @files:docs/q1-milestones.md
```

Ciri reads both files, synthesizes the key highlights and narrative, and drafts a structured update email. Approve or ask for revisions.

### Competitive Analysis

```
You > @subagents:web_researcher — Research our top 3 competitors (Notion, Linear, Asana)
      and give me a feature comparison table focused on AI features they've shipped in 2026
```

The `web_researcher` subagent crawls competitor sites, changelogs, and Product Hunt announcements, then returns a structured comparison.

### Generate Brand Assets

If you have the `canvas-design` or `brand-guidelines` skill active:

```
You > Create a set of social media banner templates for our rebrand. Brand colors:
      #6B46C1 (purple), #0BC5EA (cyan). Font: Inter. Style: minimal, modern.
```

---

## Document Processing

### Extract Tables from a PDF Report

```
You > Extract all tables from @files:reports/annual-report-2025.pdf
      and save them as a single Excel file
```

Ciri uses the `pdf` skill with `pdfplumber` to extract tables and `openpyxl` to write the Excel file. You'll see an approval prompt before any file is written.

### Merge and Reformat Word Documents

```
You > Merge all the proposal drafts in @folders:docs/proposals/ into one clean document.
      Standardize headings to Title Case and remove all tracked changes.
```

Ciri activates the `docx` skill, reads each file, merges them, and applies formatting. Writes the output to `docs/proposals/merged-final.docx` after your approval.

### Create a Presentation from Notes

```
You > Turn @files:notes/strategy-notes.md into a 12-slide PowerPoint deck.
      Use our brand colors. Executive audience, no fluff.
```

The `pptx` skill handles layout, slide structure, and styling.

---

## Research Workflows

### Deep-Dive Research with Web Researcher

```
You > Research the current regulatory landscape for AI in the EU in 2026.
      I need: key regulations, enforcement actions, compliance requirements for B2B SaaS,
      and how our competitors are responding. Produce a 3-page report.
```

Ciri delegates to `web_researcher`, which:
1. Identifies authoritative sources (EU AI Act portal, enforcement bodies, legal blogs)
2. Crawls and extracts key sections
3. Synthesizes into a structured report
4. Saves the report to `.ciri/research/eu-ai-regulation-2026.md`

### Market Analysis

```
You > Analyze the project management software market. Who are the top 10 players,
      what are their pricing models, and where are the underserved segments?
```

---

## Workspace Self-Evolution

### Build a Custom Skill

```
You > Build a skill for analyzing our weekly Salesforce CSV exports.
      It should extract pipeline health metrics and flag deals that
      haven't had activity in 14+ days.
```

Ciri delegates to `skill_builder`, which:
1. Analyzes the request and designs the playbook
2. Creates `.ciri/skills/salesforce-pipeline/SKILL.md`
3. Writes supporting reference files if needed

Once created, the skill is immediately available.

### Create a Toolkit for an Internal API

```
You > Build a toolkit for our internal metrics API at https://metrics.internal/v2/
      It needs endpoints for: daily active users, revenue by segment, and churn rate.
      API key is in METRICS_API_KEY env var.
```

`toolkit_builder` scaffolds a FastMCP server, writes the connection code, and registers it in `.ciri/toolkits/internal-metrics/`.

### Train Ciri on Your Domain

```
You > /sync
```

After any major project change — new module, rebrand, new workflow — run `/sync`. The Trainer Agent re-scans everything, updates memory files, and may propose new skills or toolkits that would help you.

---

## Thread Management

### Resume Yesterday's Work

```bash
ciri
You > /threads
> research-eu-ai-regs   (last: 2 hours ago)
  q1-report-draft       (last: yesterday)
  onboarding-review     (last: 3 days ago)

You > /switch-thread
Thread: research-eu-ai-regs
Ciri > Welcome back. We were working on the EU AI regulation report.
       The web_researcher had found 12 sources and completed sections 1-2.
       Want to continue with section 3 (compliance requirements)?
```

### Parallel Workstreams

Use separate threads for separate tasks. Each thread maintains its own state, context, and tool call history:

```bash
# Terminal 1
ciri
You > /new-thread  # Work on marketing copy

# Terminal 2 (separate session)
ciri
You > /switch-thread  # Switch to technical thread
```

---

## Keyboard Shortcuts in Action

| Shortcut | What it does |
|---|---|
| `Alt+Enter` | New line without submitting (for multi-line prompts) |
| `↑` / `↓` | Scroll through prompt history |
| `Tab` | Trigger autocomplete (after `/` or `@trigger:`) |
| `Ctrl+C` | Cancel current response |
| `Ctrl+D` | Exit Ciri |

### Multi-line prompt example

```
You > Draft a performance review for Alex based on:
[Alt+Enter]
- Q4 metrics in @files:hr/alex-q4-metrics.csv
[Alt+Enter]
- Peer feedback in @files:hr/alex-360-feedback.md
[Alt+Enter]
Keep it constructive, specific, and under 400 words.
[Enter to submit]
```
