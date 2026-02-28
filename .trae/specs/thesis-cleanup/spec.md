# Thesis Manuscript Cleanup Spec

## Why
The current thesis manuscript chapters (Chapter 2, 3, 4, and 5) contain several editorial artifacts such as duplicate titles, garbled text (likely from OCR or copy-paste), and misaligned table formatting. These issues degrade the readability and professional quality of the thesis. The user has requested a cleanup of these specific issues, excluding bibliography consolidation for now.

## What Changes
- **Chapter 2 (Problem Formulation)**: Remove the duplicate `# 第2章 ...` title block at the beginning of the file.
- **Chapter 3 (Implementation)**: Fix the garbled arrow symbol (`$ o$`) in Section 3.4 title to `$\rightarrow$`.
- **Chapter 4 (Results)**:
    - Remove line number artifacts (e.g., `60->1.57->`) in Section 4.1.4.
    - Fix alignment and missing values in Table 4-5.
- **Chapter 5 (Conclusion)**:
    - Rename/consolidate Section 5.2.2 to clarify its relationship with 5.2.1.
    - Remove line number artifacts (e.g., `58->52->`) in Section 5.3.4 and Section 5.3.1.

## Impact
- **Affected Files**:
    - `thesis_paper/manuscript_5_chapter/chapter2_problem_framework.md`
    - `thesis_paper/manuscript_5_chapter/chapter3_implementation_setup.md`
    - `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md`
    - `thesis_paper/manuscript_5_chapter/chapter5_discussion_conclusion.md`

## ADDED Requirements
### Requirement: Editorial Cleanup
The manuscript files SHALL be free of obvious copy-paste artifacts such as line numbers and duplicate headers.

#### Scenario: Clean Chapter 2
- **WHEN** reading `chapter2_problem_framework.md`
- **THEN** the file should start with a single `# 第2章` block.

#### Scenario: Clean Chapter 4
- **WHEN** reading Section 4.1.4 in `chapter4_results_verification.md`
- **THEN** the text should not contain `60->1.57->` or similar artifacts.

## MODIFIED Requirements
### Requirement: Table Formatting
Table 4-5 in Chapter 4 SHALL be properly aligned with Markdown syntax, and missing values should be represented consistently (e.g., `-` or `N/A`).

## REMOVED Requirements
N/A
