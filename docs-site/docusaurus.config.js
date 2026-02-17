// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
module.exports = {
  title: 'CIRI Copilot',
  tagline: 'Contextual Intelligent Runtime Interface - Developer docs',
  url: 'https://example.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'your-org',
  projectName: 'ciri',
  i18n: { defaultLocale: 'en', locales: ['en'] },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-org/ciri/tree/main/docs-site/',
        },
        blog: false,
        theme: { customCss: require.resolve('./src/css/custom.css') },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'CIRI',
      logo: { alt: 'CIRI Logo', src: 'img/logo.svg' },
      items: [
        { type: 'doc', docId: 'intro', position: 'left', label: 'Docs' },
        { href: 'https://github.com/your-org/ciri', label: 'GitHub', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        { title: 'Docs', items: [{ label: 'Getting Started', to: '/docs/getting-started' }] },
        { title: 'Community', items: [{ label: 'Contributing', to: '/docs/contributing' }] },
        { title: 'More', items: [{ label: 'GitHub', href: 'https://github.com/your-org/ciri' }] },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Your Org.`,
    },
    prism: { theme: lightCodeTheme, darkTheme: darkCodeTheme },
  },
};
