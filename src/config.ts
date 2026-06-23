export const siteConfig = {
  name: "Nathan Truong",
  title: "Machine Learning Engineer / Data Scientist",
  valueSentence:
    "I turn messy data into clear insights, predictive models, and better decisions.",
  description: "Portfolio website of Nathan Truong",
  accentColor: "#1d4ed8",
  social: {
    email: "nathan.truong88@gmail.com",
    linkedin: "http://www.linkedin.com/in/n-truong",
    github: "https://github.com/RyanFitzgerald",
  },
  aboutMe:
    "I'm a data scientist with 3.5 years of experience, including time as a Machine Learning Engineer at <strong>Atlassian</strong>. I focus on <strong>applied ML and AI agents</strong>, and I enjoy building solutions to complex data problems. I have a strong foundation in statistics, with a background in actuarial studies. I'm relocating from Sydney to Paris for about six months and open to <strong>part-time work, up to 3 days a week</strong>, with startups shipping AI products.",
  skills: ["Python", "SQL", "PyTorch", "LLMS/RAG", "Databricks", "AWS"],
  projects: [
    {
      name: "rag-eval — RAG Regression Harness",
      description:
        "An offline-first harness that catches RAG regressions before they ship. Scores retrieval hit@k, EM/F1, and groundedness across runs, then auto-flags any metric that drops.",
      link: "https://github.com/Nathen-T/agent_eval_harness",
      skills: ["Python", "RAG", "LLMs"],
    },
    {
      name: "Chrome Extension Mastery: Build Full-Stack Extensions with React & Node.js",
      description:
        "Master the art of building production-ready, full-stack Chrome Extensions using modern web technologies and best practices",
      link: "https://fullstackextensions.com/?ref=devportfolio",
      skills: ["React", "Node.js", "AWS"],
    },
    {
      name: "ExtensionKit",
      description:
        "Kit to jump-start your Chrome extension projects with a variety of battle-tested starter templates & examples",
      link: "https://extensionkit.io/?ref=devportfolio",
      skills: ["React", "Node.js", "AWS"],
    },
  ],
  experience: [
    {
      company: "Atlassian",
      title: "Machine Learning Engineer Intern",
      dateRange: "Nov 2025 - Feb 2026",
      bullets: [
        "Improved Confluence's personalised \"Improve Writing\" feature, used by 500K+ users.",
        "Drove an 83% gain in writing quality through prompt engineering over model and data changes.",
        "Built and validated LLM-as-a-judge metrics, then shipped the winner into a production A/B test.",
      ],
    },
    {
      company: "ITSM",
      title: "Data Scientist",
      dateRange: "Aug 2024 - Present",
      bullets: [
        "Built an ALS recommender serving 150K+ players monthly, lifting gaming activity 15%.",
        "Shipped a game-similarity tool for 20+ VIP hosts, cutting rec time from 1 hour to 10 minutes.",
        "Operated all models as automated, self-retraining pipelines with drift monitoring and alerting.",
      ],
    },
    {
      company: "WHEREFIT",
      title: "Partnerships Associate",
      dateRange: "Feb 2022 - Aug 2022",
      bullets: [
        "Owned corporate sales strategy at an early-stage startup, driving a 300% revenue increase.",
        "Lifted lead conversion 20% through A/B-tested cold-email outbound.",
      ],
    },
  ],
  education: [
    {
      school: "University of Sydney",
      degree: "Master of Data Science",
      dateRange: "2024 - 2026",
      achievements: [
        "Major in Machine Learning",
        "Coursework: Deep Learning, NLP, Statistical Learning, Big Data Engineering",
        "Capstone: XGBoost + LLM system for financial chart-pattern analysis (React/FastAPI)"
      ],
    },
    {
      school: "University of New South Wales",
      degree: "Bachelor of Actuarial Studies and Bachelor of Commerce",
      dateRange: "2019 - 2022",
      achievements: [
        "Majors: Actuarial Science and Economics",
        "Relevant: Stochastic Modelling, Statistics, Risk Theory"
      ],
    },
  ],
};
