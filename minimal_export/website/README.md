# Sparse2Full Documentation Website

This is the documentation website and landing page for the Sparse2Full project, built with React, Vite, and Tailwind CSS.

## Features
- **Responsive Design**: Mobile-first UI optimized for all devices.
- **Dark Theme (Geek Style)**: Tailored for developers.
- **i18n Support**: English and Simplified Chinese out-of-the-box.
- **Search**: Built-in fuzzy search using `fuse.js`.
- **A11y & SEO**: WCAG 2.1 AA compliant, ARIA labels, semantic HTML.
- **Security**: CSP and HSTS headers prepared for production deployment.

## Local Development

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Installation
```bash
cd website
npm install
```

### Running Locally
```bash
npm run dev
```
The app will be available at `http://localhost:5173`.

### Testing
We use Vitest for unit testing and Cypress for E2E testing.
```bash
npm run test
npm run cypress:open
```

### Load Testing
A k6 script is provided in `load-test.js` to simulate 1000 concurrent users.
```bash
k6 run load-test.js
```

## Deployment
This project is automatically deployed to GitHub Pages via GitHub Actions.
Any push to the `main` branch triggers the `.github/workflows/deploy.yml` action, which builds the static site and deploys it to the `gh-pages` branch.

### Environment Variables
- `VITE_API_URL` (optional): Set this if connecting to a real backend.
- `GA_MEASUREMENT_ID`: Replace the mock ID in `index.html` with your real Google Analytics 4 Measurement ID.