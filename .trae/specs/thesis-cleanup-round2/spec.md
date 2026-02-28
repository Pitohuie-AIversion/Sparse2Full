# Thesis Manuscript Cleanup Round 2 Spec

## Why
After the initial cleanup, three structural and editorial issues remain:
1.  **Duplicate Abstract**: The English abstract is duplicated in `chapter0_abstract.md`.
2.  **Citation Inconsistency**: Chapter 1 uses mixed citation styles (numeric, author-year, etc.), which should be standardized.
3.  **Image Path Risk**: Chapter 3 uses relative paths (`../figures_nn/...`) which are fragile compared to the standard `images/` structure used in other chapters.

## What Changes
- **Chapter 0 (Abstract)**: Remove the duplicate English abstract section.
- **Chapter 1 (Introduction)**: Standardize in-text citations to numeric format `[x]` (e.g., change `(UNESCO-IOC, 2025)` to `[9]`).
- **Chapter 3 (Implementation)**: Update image paths from relative `../figures_nn/...` to `images/...` (assuming images will be or are available there, or just standardization of the link text for now). *Note: The user's prompt implies fixing the path format itself to match the standard.*

## Impact
- **Affected Files**:
    - `thesis_paper/manuscript_5_chapter/chapter0_abstract.md`
    - `thesis_paper/manuscript_5_chapter/chapter1_intro_related.md`
    - `thesis_paper/manuscript_5_chapter/chapter3_implementation_setup.md`

## ADDED Requirements
### Requirement: Content Uniqueness
The abstract file SHALL contain only one instance of the English abstract.

## MODIFIED Requirements
### Requirement: Citation Format
In-text citations in Chapter 1 SHALL use the numeric `[x]` format consistently.

### Requirement: Image Path Consistency
Image references in Chapter 3 SHALL use the `images/` directory prefix to match other chapters.

## REMOVED Requirements
- Redundant English Abstract section in Chapter 0.
