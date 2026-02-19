module.exports = {
  tutorialSidebar: [
    'index',
    'getting-started',
    {
      type: 'category',
      label: 'CLI Reference',
      collapsed: false,
      items: [
        'cli-reference',
        'examples',
      ],
    },
    {
      type: 'category',
      label: 'Features',
      collapsed: false,
      items: [
        'features/memory',
        'features/hitl',
        'features/autocomplete',
        'features/web-research',
        'features/script-executor',
      ],
    },
    {
      type: 'category',
      label: 'Skills',
      collapsed: false,
      items: [
        'built-in-skills/index',
        'skills-guide',
      ],
    },
    {
      type: 'category',
      label: 'Toolkits & Subagents',
      items: [
        'toolkits-guide',
        'subagents-guide',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/core-harness',
      ],
    },
    {
      type: 'category',
      label: 'Internals',
      items: [
        'internals/controller',
        'internals/middlewares',
        'internals/self-evolution',
        'internals/graph-execution',
      ],
    },
    {
      type: 'category',
      label: 'Project',
      items: [
        'contributing',
        'security',
        'faq',
      ],
    },
  ],
};
