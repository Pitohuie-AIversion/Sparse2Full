# Thesis Figure Verification Spec

## Why
A comprehensive check of image references in the thesis manuscript revealed that while most figures are present in the `images/` directory, **Chapter 3** contains broken links. Specifically, `fig_edsr_auto.svg` and `fig_videoswin_auto.svg` were updated to point to `images/` in the previous cleanup, but the actual files were not moved from their original source directory (`figures_nn/...`). This causes missing figures in Chapter 3.

## What Changes
- **File System**: Copy the missing SVG files from the `figures_nn` build directory to the `manuscript_5_chapter/images` directory.
    - Source: `thesis_paper/figures_nn/build_export_j2/edsr/fig_edsr_auto.svg` -> Dest: `thesis_paper/manuscript_5_chapter/images/fig_edsr_auto.svg`
    - Source: `thesis_paper/figures_nn/build_export_j2/videoswin/fig_videoswin_auto.svg` -> Dest: `thesis_paper/manuscript_5_chapter/images/fig_videoswin_auto.svg`

## Impact
- **Affected Files**: None (text remains unchanged).
- **Affected Assets**: `thesis_paper/manuscript_5_chapter/images/` will now contain the required SVG files.

## ADDED Requirements
### Requirement: Figure Availability
All figures referenced in the markdown files SHALL exist in the relative `images/` path to ensure the manuscript is self-contained.

## MODIFIED Requirements
N/A

## REMOVED Requirements
N/A
