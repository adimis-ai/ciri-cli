// @ts-check
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
module.exports = {
  title: 'Ciri',
  tagline: 'Your local AI copilot — trained on your workspace, built for any domain.',
  url: 'https://ciri.adimis.in',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'Adimis',
  projectName: 'ciri',
  i18n: { defaultLocale: 'en', locales: ['en'] },
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/adimis-ai/ciri/tree/main/docs-site/',
        },
        blog: false,
        theme: { customCss: require.resolve('./src/css/custom.css') },
      },
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'Ciri',
      items: [
        {
          type: 'doc',
          docId: 'index',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'doc',
          docId: 'getting-started',
          position: 'left',
          label: 'Getting Started',
        },
        {
          type: 'doc',
          docId: 'built-in-skills/index',
          position: 'left',
          label: 'Skills',
        },
        {
          type: 'doc',
          docId: 'roadmap',
          position: 'left',
          label: 'Roadmap',
        },
        {
          href: 'https://github.com/adimis-ai/ciri',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Get Started',
          items: [
            { label: 'Introduction', to: '/docs/' },
            { label: 'Installation', to: '/docs/getting-started' },
            { label: 'CLI Reference', to: '/docs/cli-reference' },
            { label: 'Examples', to: '/docs/examples' },
          ],
        },
        {
          title: 'Features',
          items: [
            { label: 'Workspace Memory', to: '/docs/features/memory' },
            { label: 'Human-in-the-Loop', to: '/docs/features/hitl' },
            { label: 'Web Research', to: '/docs/features/web-research' },
            { label: 'Script Executor', to: '/docs/features/script-executor' },
          ],
        },
        {
          title: 'Extend Ciri',
          items: [
            { label: 'Built-in Skills', to: '/docs/built-in-skills' },
            { label: 'Skills Guide', to: '/docs/skills-guide' },
            { label: 'Toolkits Guide', to: '/docs/toolkits-guide' },
            { label: 'Subagents Guide', to: '/docs/subagents-guide' },
          ],
        },
        {
          title: 'Project',
          items: [
            { label: 'Roadmap', to: '/docs/roadmap' },
            { label: 'Contributing', to: '/docs/contributing' },
            { label: 'Security', to: '/docs/security' },
            { label: 'FAQ', to: '/docs/faq' },
            { label: 'GitHub', href: 'https://github.com/adimis-ai/ciri' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Adimis. Built with Docusaurus.`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['bash', 'python', 'javascript', 'typescript', 'yaml', 'json'],
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  },
};
