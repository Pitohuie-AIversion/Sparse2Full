# Thesis Figure Numbering & Consistency Spec

## Why
A review of the thesis manuscript chapters revealed inconsistencies in figure references and numbering:
1.  **Chapter 1 (Intro)**: Line 359 incorrectly references `图 1-3` (which is already used in line 161 for a different figure), while the text context ("本文的整体组织结构与逻辑流程") and the adjacent figure placeholder (Line 361) clearly correspond to `图 1-5`.
2.  **Chapter 5 (Conclusion)**: Figure numbering starts at `图 5-2` (Section 5.3.1), skipping `图 5-1`. This suggests a missing figure or, more likely, an indexing error where the sequence should start from `5-1`.

## What Changes
- **Chapter 1**: Correct the text reference `如图 1-3 所示` to `如图 1-5 所示` in Section 1.4.
- **Chapter 5**: Renumber figures `5-2`, `5-3`, `5-4` to `5-1`, `5-2`, `5-3` respectively to ensure sequential numbering.

## Impact
- **Affected Files**:
    - `thesis_paper/manuscript_5_chapter/chapter1_intro_related.md`
    - `thesis_paper/manuscript_5_chapter/chapter5_discussion_conclusion.md`

## ADDED Requirements
### Requirement: Sequential Figure Numbering
Figures within each chapter SHALL be numbered sequentially starting from X-1 (e.g., 5-1, 5-2, ...).

## MODIFIED Requirements
### Requirement: Consistent References
Text references to figures SHALL match the figure label and content being discussed.

## REMOVED Requirements
N/A
