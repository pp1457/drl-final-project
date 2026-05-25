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

## OverLeaf checklist
1. Upload **both** `main.tex` AND `refs.bib` to the project root.
2. Also upload `neurips_2026.sty` (from the official author kit or by
   creating from the NeurIPS 2026 template).
3. Provide `training_curve.png` for Figure 1 (or comment out the
   `\includegraphics` line if you do not have it).
4. Compile with pdfLaTeX. OverLeaf will auto-detect the
   `\bibliography{refs}` and run bibtex.

## Status of sections (final v10c version)
| Section | Status |
|---|---|
| Abstract | Final |
| 1. Introduction | Final |
| 2. Related Work | Final (11 refs, all bibtex-wired) |
| 3. Method | Final (env, PPO, OCA, DPR, charge-dim, RND, orchestration) |
| 4. Experiments | Final (training curves, per-seed, eval, RND-collapse, stability) |
| 5. Discussion | Final |
| 6. Systems-level variants | Final (capture-latency comparison A/B/C) |
| 7. Limitations & Future Work | Final |
| 8. Conclusion | Final (closes loop with Q1/Q2/Q3 from intro) |
| Appendix A | Hyperparameters (PPO, RND, aux, obs, eval) |
| Appendix B | Per-update checkpoint schedule |
| Appendix C | Reproducibility (exact CLI invocations) |
