# Paper draft

Skeleton LaTeX source for the NTU-DRL-MiniConf 2026 submission.

## Files
- `main.tex` — paper skeleton (full structure, TODO markers in places needing data)
- `refs.bib` — bibliography with the 11 key references identified

## Compiling

### OverLeaf (recommended)
1. Create new project → upload `main.tex` + `refs.bib`
2. In the OverLeaf project, click "Menu" → "Settings" → choose "pdfLaTeX"
3. **Important**: add `neurips_2026.sty` (download from the official NeurIPS 2026 author kit).
   Alternatively use OverLeaf's "New Project from Template" → "NeurIPS 2026" and copy `main.tex` contents into the template.
4. Build.

### Local
```
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

`neurips_2026.sty` must be either in this directory or on `TEXINPUTS`.

## Anonymity check
- Authors block reads "Anonymous Authors" — DO NOT change for review submission
- Line numbers are visible (default behavior without the `final` option)
- No external links revealing identity (e.g., no GitHub repo URL to your account)
- `\todo{...}` markers are colored red for drafting; remove before submission

## Status of sections
| Section | Status |
|---|---|
| Abstract | Drafted, needs final numbers |
| 1. Introduction | Drafted, may need polish |
| 2. Related Work | Drafted with 11 refs |
| 3. Method | Method sub-sections drafted; some math TODO |
| 4. Environment & Diagnostic Methodology | Skeleton only; needs writing |
| 5. Experiments | Placeholder table; needs writing once eval lands |
| 6. Discussion | Empty; write last |
| 7. Conclusion | Empty; write last |
| Appendices A-D | Empty; pull from LESSONS_LEARNED.md, hyperparams, logs |
