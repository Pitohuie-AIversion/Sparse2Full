# Pixel-Perfect QA Checklist

Use this checklist to verify that the implementation matches the design system specifications.

## 1. Environment Setup
- [ ] **100% Zoom**: Ensure browser zoom is reset to 100% (Cmd+0 / Ctrl+0).
- [ ] **Retina Display**: Verify on a high-density display (2x/3x) if possible, or simulate via DevTools.
- [ ] **Tools Installed**:
  - [PerfectPixel](https://www.welldonecode.com/perfectpixel/) (Chrome Extension) or similar overlay tool.
  - [VisBug](https://chrome.google.com/webstore/detail/visbug/cdockenadnadldjbbgcallicgledbeoc) (for quick measurements).
  - [Axe DevTools](https://www.deque.com/axe/devtools/) (for accessibility).

## 2. Grid & Layout Verification
- [ ] **Container Max-Widths**:
  - Mobile (<768px): Fluid width, `16px` padding.
  - Tablet (≥768px): `720px` max-width, `24px` gutter.
  - Desktop (≥1024px): `960px` max-width, `24px` gutter.
  - Wide (≥1440px): `1320px` max-width, `32px` gutter.
- [ ] **Column Alignment**: Overlay the grid system image. Content must align perfectly with column start/end lines.
- [ ] **Gutter Widths**: Measure gaps between columns. Must match `16px` (Mobile), `24px` (Tablet/Desktop), `32px` (Wide).

## 3. Spacing Audit
- [ ] **Vertical Rhythm**: All vertical margins/paddings must be multiples of **4px** or **8px**.
- [ ] **Component Padding**:
  - Buttons: `12px` (sm), `16px` (md), `24px` (lg).
  - Inputs: `12px` horizontal.
  - Cards: `24px` padding.
- [ ] **Text Spacing**: Check `line-height` creates proper vertical rhythm.

## 4. Typography Check
- [ ] **Font Family**: Computed style must show `Inter` (Body) or `Space Grotesk` (Headings).
- [ ] **Font Sizes**:
  - H1: `48px` (Mobile) -> `60px` (Desktop).
  - H2: `36px` (Mobile) -> `48px` (Desktop).
  - Body: `16px`.
  - Small: `14px`.
- [ ] **Line Heights**: Verify calculated pixel values (e.g., `16px` font * `1.5` line-height = `24px` computed).

## 5. Component Fidelity
- [ ] **Buttons**:
  - Height: `32px` (sm), `40px` (md), `48px` (lg).
  - Border Radius: `4px` exactly.
- [ ] **Inputs**:
  - Height: `32px` (sm), `40px` (md), `48px` (lg).
  - Border Color: `#E2E8F0` (default), `#3B82F6` (focus).
- [ ] **Shadows**: Verify `box-shadow` values match tokens (no default browser shadows).

## 6. Responsive Behavior
- [ ] **Breakpoint 768px**: Layout shifts from 4-col to 8-col.
- [ ] **Breakpoint 1024px**: Layout shifts from 8-col to 12-col.
- [ ] **Touch Targets**: All interactive elements must be at least `44x44px` on mobile (check computed hit area).

## 7. Automated Testing (Optional)
- [ ] **Snapshot Testing**: Run `npm test` to check Jest/Storybook snapshots.
- [ ] **Visual Regression**: Run BackstopJS or Percy tests to detect pixel shifts.

## 8. Accessibility (A11y)
- [ ] **Color Contrast**: Text must meet WCAG AA standards (4.5:1 ratio).
- [ ] **Focus States**: All interactive elements must have a visible focus ring.
