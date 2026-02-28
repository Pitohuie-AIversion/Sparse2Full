# Pixel-Perfect Design System

This directory contains the core design system for the project, providing a unified standard for pixel-level layout, typography, and components.

## 1. Structure

- `design_tokens.json`: **Single Source of Truth**. Contains all raw values (colors, spacing, typography, grid).
- `styles/`: SCSS implementation.
  - `_variables.scss`: SCSS variables and CSS custom properties mapping.
  - `_mixins.scss`: Helper functions for responsiveness and typography.
  - `_grid.scss`: The 12-column grid system implementation.
  - `_typography.scss`: Typography scale and global styles.
  - `_components.scss`: Core component styles (Buttons, Inputs, Cards).
- `qa_checklist.md`: Checklist for verifying implementation accuracy.

## 2. Usage in Code

Import the styles in your main SCSS file:

```scss
// Import variables first
@use "design_system/styles/variables" as *;
@use "design_system/styles/mixins" as *;

// Import core styles
@use "design_system/styles/grid";
@use "design_system/styles/typography";
@use "design_system/styles/components";

// Use variables in your custom components
.my-component {
  padding: $spacing-md; // 16px
  background-color: $color-surface;
  border-radius: $card-radius;
  
  @include media-up(tablet) {
    padding: $spacing-lg; // 24px
  }
}
```

## 3. Usage in Design Tools (Figma/Sketch)

The `design_tokens.json` file is formatted to be compatible with modern design tools.

### Figma
1. Install the **[Tokens Studio for Figma](https://tokens.studio/)** plugin.
2. Open the plugin in your Figma file.
3. Go to the "Tools" or "Settings" tab.
4. Select "Load from JSON" or "Import".
5. Paste the content of `design_tokens.json`.
6. This will automatically generate:
   - Color Styles
   - Text Styles
   - Spacing Tokens
   - Border Radius Tokens

### Sketch
1. Use a plugin like **[JSON to Sketch](https://github.com/sketch-hq/sketch-json-podcast)** or a similar token importer.
2. Map the JSON keys to Sketch Layer Styles and Text Styles.

## 4. Grid System Standards

- **Mobile (<768px)**: 4 Columns, 16px Gutter, 16px Margin.
- **Tablet (≥768px)**: 8 Columns, 24px Gutter, 32px Margin.
- **Desktop (≥1024px)**: 12 Columns, 24px Gutter, 64px Margin.
- **Wide (≥1440px)**: 12 Columns, 32px Gutter, Auto Margin (Max-width 1320px).

## 5. QA Process

Refer to `qa_checklist.md` for the step-by-step verification process to ensuring pixel-perfect implementation.
