# Fix Thesis Manuscript Spec

## Why
A comprehensive quality assurance check revealed high-risk formatting and content issues in the thesis manuscript (Chapters 1-5). These include inconsistent citation formats (mixed GB/T 7714 and APA), residual draft artifacts (draft titles, AI generation glitches), and inappropriate content (modification logs in main text). These must be resolved to meet submission standards.

## What Changes
- **Chapter 1 (Introduction)**: 
  - Remove draft title line `# 一、1.1.2 可直接替换的正式版本...`.
  - Fix garbled citation `【6†source:contentReference...`.
- **Chapter 4 (Experiments)**:
  - Convert APA style references to GB/T 7714-2015 numeric style.
  - Remove `Appendix: Modification Log` section.
- **Chapter 5 (Conclusion)**:
  - Convert APA style references to GB/T 7714-2015 numeric style.

## Impact
- **Affected Files**:
  - `thesis_paper/manuscript_5_chapter/chapter1_intro_related.md`
  - `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md`
  - `thesis_paper/manuscript_5_chapter/chapter5_discussion_conclusion.md`

## ADDED Requirements
N/A

## MODIFIED Requirements
### Requirement: Consistent Citation Format
The thesis SHALL use **GB/T 7714-2015** (numeric `[1]`) citation style consistently across all chapters.

### Requirement: Clean Manuscript
The manuscript SHALL be free of draft artifacts, temporary headings, and internal revision logs.

## REMOVED Requirements
### Requirement: Modification Logs
**Reason**: Internal modification logs are for tracking changes and should not appear in the final thesis manuscript.
