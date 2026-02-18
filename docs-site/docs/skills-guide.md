# Skills Guide

Skills are the fundamental building blocks of Ciri's domain knowledge. A skill is a **self-contained package** consisting of a `SKILL.md` playbook and optional Python helper scripts. Skills tell Ciri *when* to activate and *how* to approach tasks in a specific domain — whether that domain is PDF processing, financial reporting, marketing copy, or legal document review.

---

## What Is a Skill?

A skill is not a plugin or tool — it is a **domain playbook**. It provides:

1. **A routing trigger** — the `description` field in the SKILL.md frontmatter tells Ciri when to use this skill. Written from Ciri's perspective: "Use me when the user mentions `.pdf` files or wants to work with a PDF."
2. **A procedural playbook** — the body of SKILL.md contains step-by-step instructions, code examples, critical rules, and decision trees Ciri follows for that domain.
3. **Helper scripts** — the `scripts/` directory holds Python utilities that Ciri can invoke via the `execute_script` tool. These handle complex, repeatable operations without loading full source code into context.

---

## Skill Directory Structure

```
<skill-name>/
├── SKILL.md           ← Required: frontmatter trigger + playbook
├── LICENSE.txt        ← Optional but recommended
├── scripts/           ← Optional executable Python helpers
│   ├── process.py
│   └── validate.py
├── references/        ← Optional reference docs loaded on demand
│   └── advanced.md
└── assets/            ← Optional static files (templates, fonts)
    └── template.docx
```

Skills live in one of two locations:

| Location | Scope |
|---|---|
| `~/.local/share/ciri/skills/` | All projects (global core harness) |
| `.ciri/skills/` | This project only (project harness) |

→ [Two-Level Harness Architecture](architecture/core-harness.md)

---

## SKILL.md Frontmatter

The frontmatter is what the `SkillsMiddleware` reads to understand and route to a skill:

```yaml
---
name: quarterly-reporting
description: |
  Use this skill when the user wants to produce, review, or analyze quarterly
  financial reports. Triggers include: P&L analysis, income statement, EBITDA,
  revenue breakdown, YoY comparison, board report, investor update.
  Also trigger when the user references internal reporting templates or asks
  to format financial data for a presentation.
  Do NOT trigger for general Excel tasks, non-financial documents, or coding work.
license: Proprietary
---
```

**Writing a good description:**
- Be specific about trigger phrases
- Include "Do NOT trigger" exclusions to prevent false positives
- Write as if explaining to Ciri, not to a human reader
- Match actual user language ("deck", "slides" — not "presentation document")

---

## SKILL.md Body

The body is the playbook Ciri reads when the skill activates. Structure it for rapid utility:

```markdown
## Quick Reference

| Task | Approach |
|---|---|
| Create P&L from scratch | Use `scripts/generate_pl.py` |
| Compare to prior period | Load both sheets, compute YoY delta |

## Standard Workflow

1. Read the source data file
2. Validate against the template schema (see `references/schema.md`)
3. Apply formatting rules from `assets/brand-template.xlsx`

## Critical Rules

- Never use placeholder data — always flag missing values explicitly
- Always include a variance column vs. prior period
- Round to 2 decimal places for currency, 1 for percentages

## Common Mistakes

**DON'T** sum rows manually — use formulas so the model recalculates
**DO** use standard color coding: green = favorable, red = unfavorable
```

**Playbook best practices:**
- Keep SKILL.md under 500 lines — overflow detail into `references/`
- Use Quick Reference tables for the most common patterns
- "Critical Rules" and "Common Mistakes" sections prevent regressions
- Use actual code examples, not pseudocode

---

## Creation Workflows

### 1. Ask Ciri (recommended)

```
You > Create a skill for our Salesforce CRM reporting workflow
You > Add a skill for processing government procurement documents
You > Build a skill for our internal brand voice and copy guidelines
You > Create a skill to handle our quarterly investor deck format
```

Ciri invokes the `skill_builder_agent`, which uses the `skill-creator` meta-skill to scaffold and write a complete skill. It is placed in the core harness (globally available) unless you ask for project-specific placement.

### 2. `/sync` with domain focus

```
You > /sync — focus on adding capabilities for our data warehouse (Snowflake + dbt)
You > /sync — I need Ciri to understand our legal document review process
```

The Trainer Agent runs its **AUDIT → ANALYZE → PLAN → BUILD → VERIFY** loop and creates targeted skills.

### 3. Manual creation

```bash
mkdir -p .ciri/skills/my-domain

cat > .ciri/skills/my-domain/SKILL.md << 'EOF'
---
name: my-domain
description: Use this skill when the user asks about [your domain and specific triggers].
---

# My Domain Guide

## Approach
[Step-by-step playbook for this domain]
EOF
```

Then run `/sync` to hot-reload.

---

## Skill Dependencies

If your helper scripts require Python packages, add a `requirements.txt`:

```
# .ciri/skills/my-skill/requirements.txt
pandas>=2.0
openpyxl
requests
```

Ciri installs them via `uv pip install` when the script first runs.

---

## Hot-Reload

Skills are discovered dynamically — no restart needed. After creating or modifying a skill, run:

```
You > /sync
```

---

## Built-in Skills (17 Included)

Ciri ships with 17 built-in skills across documents, design, engineering, and communications, bootstrapped to `~/.local/share/ciri/skills/` on first run.

→ [Full Built-in Skills Catalog](built-in-skills/index.md)

---

## Skill vs. Toolkit vs. SubAgent

| Component | Purpose | Create When |
|---|---|---|
| **Skill** | Domain playbook + scripts | You need specialized knowledge, a workflow, or recurring patterns |
| **Toolkit** | MCP server for an external service | You need to call an external API or database (Slack, Jira, Postgres) |
| **SubAgent** | Specialized agent with its own scope | You need a different persona, tool set, or reasoning approach |

Ask Ciri if unsure: "Should I create a skill, toolkit, or subagent for [X]?"
