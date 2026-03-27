# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2023-10-25
### Added
- Multi-page routing (`react-router-dom`) with Home, Docs, About, Contact, Login, and Admin pages.
- Internationalization (`react-i18next`) with English and Chinese support.
- Accessibility (A11y) improvements including ARIA labels across components.
- Security headers (CSP, HSTS) and GA4 analytics mock in `index.html`.
- Testing setup with Vitest and Cypress, including a basic test suite.
- Load testing template (`load-test.js`) using `k6`.

### Changed
- Refactored `App.jsx` to serve as a router outlet.
- Moved original single-page documentation content to `Docs.jsx`.

## [1.0.0] - Initial Release
- Setup Vite + React + Tailwind CSS project.
- Implement responsive dark-theme design.
- Create single-page documentation with `react-syntax-highlighter`.
- Implement global search using `fuse.js`.