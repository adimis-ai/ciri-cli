import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const features = [
  {
    icon: '🧠',
    title: 'Domain-Adaptive Intelligence',
    description:
      'Ciri trains herself to match your workspace. Marketing agency, fintech startup, law firm — run /sync and she learns your domain, tools, and workflows.',
  },
  {
    icon: '♾️',
    title: 'Self-Evolving Capabilities',
    description:
      'Permanently expands her own toolset by creating new Skills, Toolkits, and SubAgents. The more you work with Ciri, the more capable she becomes.',
  },
  {
    icon: '🌐',
    title: 'Real Browser Integration',
    description:
      'Connects to Chrome/Edge via CDP using your real sessions and cookies. Research inside your AWS console, GitHub, CRM, or any authenticated app.',
  },
  {
    icon: '🔒',
    title: 'Human-in-the-Loop',
    description:
      'Every file write, script execution, or external action requires your explicit approval. Approve, edit the arguments, or reject — you stay in control.',
  },
  {
    icon: '🔌',
    title: 'MCP Toolkit Ecosystem',
    description:
      'Native Model Context Protocol client. Connect to any MCP server — Slack, databases, APIs, cloud services — and hot-reload without restart.',
  },
  {
    icon: '🗄️',
    title: 'Local-First & Private',
    description:
      'Conversation threads, checkpoints, and workspace memory live in a local SQLite database.',
  },
];

const useCaseRows = [
  { domain: 'Software Engineering', examples: 'Read codebases · write & test code · manage git · generate PRs · run CI' },
  { domain: 'Business Operations', examples: 'Draft proposals · analyze financials · automate reporting · manage trackers' },
  { domain: 'Marketing & Branding', examples: 'Research competitors · draft copy · analyze engagement · schedule campaigns' },
  { domain: 'Sales & CRM', examples: 'Qualify leads · draft outreach · summarize calls · update CRM via MCP' },
  { domain: 'Research & Data', examples: 'Crawl the web authenticated · summarize papers · build data pipelines' },
  { domain: 'Document Work', examples: 'Fill PDF forms · process DOCX/PPTX/XLSX · extract structured data' },
];

const steps = [
  { step: '01', title: 'Install', code: 'pip install ciri-copilot' },
  { step: '02', title: 'Run', code: 'ciri' },
  { step: '03', title: 'Sync', code: '/sync  # Ciri learns your workspace' },
  { step: '04', title: 'Evolve', code: '# Ask anything. Ciri grows with you.' },
];

export default function Home() {
  return (
    <Layout
      title="CIRI — Contextual Intelligence and Reasoning Interface"
      description="Ciri is an autonomous, self-evolving AI copilot that adapts to any domain — software, marketing, finance, operations, research, and beyond."
    >
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <header className="hero-banner">
        <div className="container">
          <div style={{ marginBottom: '1.5rem' }}>
            <span
              style={{
                display: 'inline-block',
                background: 'rgba(139,92,246,0.15)',
                border: '1px solid rgba(139,92,246,0.4)',
                borderRadius: '99px',
                padding: '0.4rem 1.2rem',
                fontSize: '0.85rem',
                color: 'var(--ifm-color-primary-light)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                fontWeight: 600,
              }}
            >
              Open Source · Local-First · Self-Evolving
            </span>
          </div>
          <h1 className="hero-title">Your AI Copilot.<br />Any Domain.</h1>
          <p className="hero-subtitle">
            Ciri is an autonomous AI copilot that lives inside your workspace and adapts to
            your domain — software engineering, marketing, finance, operations, research, and beyond.
            She builds a persistent understanding of your environment and permanently expands
            her own capabilities over time.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link className="button-primary" to="/docs/getting-started">
              Get Started →
            </Link>
            <Link
              className="button-primary"
              style={{ background: 'transparent', border: '1px solid rgba(139,92,246,0.5)' }}
              to="/docs/intro"
            >
              Read the Docs
            </Link>
          </div>

          {/* Terminal preview */}
          <div
            style={{
              marginTop: '4rem',
              background: 'rgba(2, 6, 23, 0.8)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '1rem',
              padding: '1.5rem 2rem',
              maxWidth: '680px',
              margin: '4rem auto 0',
              textAlign: 'left',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              lineHeight: 1.8,
            }}
          >
            <div style={{ color: 'rgba(255,255,255,0.3)', marginBottom: '0.75rem', fontSize: '0.75rem' }}>
              ● ● ●
            </div>
            <div><span style={{ color: '#8b5cf6' }}>$</span> <span style={{ color: '#e2e8f0' }}>ciri</span></div>
            <div style={{ color: '#64748b' }}>  ██████╗ ██╗ ██████╗  ██╗</div>
            <div style={{ color: '#64748b' }}>  ██╔════╝ ██║ ██╔══██╗ ██║</div>
            <div style={{ color: '#64748b' }}>  ██║      ██║ ██████╔╝ ██║</div>
            <div style={{ color: '#64748b' }}>  ██║      ██║ ██╔══██╗ ██║</div>
            <div style={{ color: '#64748b' }}>  ╚██████╗ ██║ ██║  ██║ ██║</div>
            <div style={{ color: '#64748b' }}>   ╚═════╝ ╚═╝ ╚═╝  ╚═╝ ╚═╝</div>
            <div style={{ color: '#a78bfa', marginTop: '0.5rem' }}>  Contextual Intelligence and Reasoning Interface</div>
            <div style={{ color: '#22d3ee', marginTop: '0.75rem' }}>✓ Playwright ready</div>
            <div style={{ color: '#22d3ee' }}>✓ Model: claude-sonnet-4-6</div>
            <div style={{ color: '#22d3ee' }}>✓ Database initialized</div>
            <div style={{ color: '#22d3ee' }}>✓ Workspace: ~/projects/my-startup</div>
            <div style={{ color: '#22d3ee' }}>✓ 12 skills · 3 toolkits · 5 subagents</div>
            <div style={{ color: '#94a3b8', marginTop: '0.75rem' }}>You &gt; <span style={{ color: '#e2e8f0' }}>Draft a Q1 investor update based on our metrics dashboard</span></div>
            <div style={{ color: '#a78bfa', marginTop: '0.5rem' }}>Ciri &gt; <span style={{ color: '#94a3b8' }}>Delegating to web_researcher to pull live metrics...</span></div>
          </div>
        </div>
      </header>

      <main>
        {/* ── Feature Grid ──────────────────────────────────────────────── */}
        <section>
          <div className="feature-grid">
            {features.map(({ icon, title, description }) => (
              <div className="feature-card" key={title}>
                <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>{icon}</div>
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Use Cases ─────────────────────────────────────────────────── */}
        <section
          style={{
            padding: '5rem 2rem',
            background: 'rgba(255,255,255,0.01)',
            borderTop: '1px solid rgba(255,255,255,0.04)',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
          }}
        >
          <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            <h2 style={{ textAlign: 'center', fontSize: '2.5rem', marginBottom: '0.75rem' }}>
              Built for Every Domain
            </h2>
            <p style={{ textAlign: 'center', color: 'var(--ifm-color-emphasis-600)', marginBottom: '3rem', fontSize: '1.1rem' }}>
              Ciri adapts to your workspace through training. The /sync command turns her into an expert for your field.
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '0.75rem 1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--ifm-color-primary-light)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Domain</th>
                    <th style={{ textAlign: 'left', padding: '0.75rem 1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--ifm-color-primary-light)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>What Ciri Can Do</th>
                  </tr>
                </thead>
                <tbody>
                  {useCaseRows.map(({ domain, examples }, i) => (
                    <tr key={domain} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                      <td style={{ padding: '0.9rem 1rem', borderBottom: '1px solid rgba(255,255,255,0.04)', fontWeight: 600, color: '#e2e8f0', whiteSpace: 'nowrap' }}>{domain}</td>
                      <td style={{ padding: '0.9rem 1rem', borderBottom: '1px solid rgba(255,255,255,0.04)', color: 'var(--ifm-color-emphasis-600)' }}>{examples}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── How It Works ──────────────────────────────────────────────── */}
        <section style={{ padding: '5rem 2rem' }}>
          <div style={{ maxWidth: '900px', margin: '0 auto', textAlign: 'center' }}>
            <h2 style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>Get Running in Minutes</h2>
            <p style={{ color: 'var(--ifm-color-emphasis-600)', marginBottom: '3rem', fontSize: '1.1rem' }}>
              Install, run, sync. Ciri does the rest.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', textAlign: 'left' }}>
              {steps.map(({ step, title, code }) => (
                <div
                  key={step}
                  style={{
                    padding: '1.75rem',
                    background: 'var(--ciri-glass)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: '1.25rem',
                  }}
                >
                  <div style={{ color: 'var(--ifm-color-primary)', fontWeight: 700, fontSize: '0.75rem', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>{step}</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.75rem' }}>{title}</div>
                  <code style={{ display: 'block', fontSize: '0.8rem', color: '#a78bfa', background: 'rgba(0,0,0,0.3)', padding: '0.6rem 0.8rem', borderRadius: '0.5rem' }}>{code}</code>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Self-Evolution Callout ─────────────────────────────────────── */}
        <section
          style={{
            padding: '5rem 2rem',
            background: 'radial-gradient(circle at 50% 50%, rgba(139,92,246,0.08) 0%, transparent 70%)',
            borderTop: '1px solid rgba(255,255,255,0.04)',
          }}
        >
          <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚡</div>
            <h2 style={{ fontSize: '2.25rem', marginBottom: '1rem' }}>Ciri Builds Her Own Tools</h2>
            <p style={{ color: 'var(--ifm-color-emphasis-600)', fontSize: '1.15rem', lineHeight: 1.7, marginBottom: '2rem' }}>
              Most AI assistants are frozen at their training cutoff. Ciri is different: when she encounters a gap in
              her capabilities, she creates new Skills (domain playbooks), Toolkits (MCP API connectors), or SubAgents
              (specialized role agents) — and loads them immediately. Your Ciri becomes uniquely yours.
            </p>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              <Link className="button-primary" to="/docs/internals/self-evolution">
                How Self-Evolution Works
              </Link>
              <Link
                className="button-primary"
                style={{ background: 'transparent', border: '1px solid rgba(139,92,246,0.5)' }}
                to="/docs/skills-guide"
              >
                Skills Guide
              </Link>
            </div>
          </div>
        </section>

        {/* ── CTA ───────────────────────────────────────────────────────── */}
        <section style={{ padding: '6rem 2rem', textAlign: 'center' }}>
          <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Ready to evolve your workspace?</h2>
          <p style={{ marginBottom: '2.5rem', color: 'var(--ifm-color-emphasis-600)', fontSize: '1.1rem' }}>
            Open source. Local-first. Works with OpenAI, Anthropic, OpenRouter, and more.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link className="button-primary" to="/docs/getting-started">
              Install Ciri →
            </Link>
            <a
              className="button-primary"
              style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.15)' }}
              href="https://github.com/adimis-ai/ciri"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub ↗
            </a>
          </div>
        </section>
      </main>
    </Layout>
  );
}
