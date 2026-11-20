# Pre-submission check — revision_v2 (2026-11-20)

Manuscript: `paper/revision_v2/manuscript_revised.tex` ("A Survey of Parameter-Efficient Fine-Tuning for Generalizable Deepfake Detection"), response letter `paper/revision_v2/response_to_reviewers.tex`.
Checklists used: `latex-document-skill` (two-column float rules, LaTeX patterns), `academic-research-skills` → academic-paper-reviewer (7 dimensions, `review_criteria_framework.md` §2.3 literature review), academic-paper (`writing_quality_check.md`, `abstract_writing_guide.md`, `journal_submission_guide.md`). Judgements are criterion-bound and NOT_CALIBRATED, as the rubric requires.

## A. LaTeX / format

| Check | Result |
|---|---|
| Compiles (tectonic), unresolved refs/cites | OK: 0 `??`, every `\cite` has a `\bibitem`, no uncited entries |
| Overfull boxes | Manuscript: 0 hbox (taxonomy table fixed); letter: 1 hbox of 11pt on the title line |
| Float rules for two columns | OK: no `[H]`, all `[!t]`; 6 full-width floats, the rest single-column; every page mixes text and floats |
| Raw `<`/`>` in text, em-dash overuse | None |
| Abstract length | 202 words (guide: 150–250) |
| Keywords | 7 (guide: 5–7), changed from 8 |
| Figures | Fig. 1–2 vector PDF from Figma, Fig. 3–4 generated vector PDF; all referenced in text |
| Appendix numbering | Table A1–A3 |
| skill scripts | `validate_latex.py` only reports "batch file" structure errors (meant for PDF-to-LaTeX batches, not applicable); `latex_lint.sh`, `latex_wordcount.sh` and `latex_package_check.sh` need system TeX Live (chktex/detex), which is not installed. They were not run |

## B. Review dimensions (academic-paper-reviewer rubric)

| Dimension | Judgement | Evidence / reason |
|---|---|---|
| Originality | MEETS | Controlled benchmark and the checkpoint-selection analysis are new relative to cited surveys; LN-vs-bias finding is new |
| Methodological rigor | PARTLY_MEETS | Clean protocol, 10/6/3 seeds, Holm + TOST. But: one backbone and one training set; GenD n=6 vs 10; LoRA/Effort n=3; the selection-rule effect (0.934 vs 0.923) is one model |
| Evidence sufficiency | PARTLY_MEETS | The additive branch (adapters/prompts) is in the taxonomy but has no benchmarked representative. RQ3 covers face-swap only (diffusion absent, stated as a limitation) |
| Argument coherence | MEETS | RQ1–RQ3 → results → analysis → recommendations trace cleanly; claims softened to what Holm-corrected tests support |
| Writing quality | MEETS | Concise; terminology consistent; flagged-term scan clean |
| Literature integration — survey criteria (§2.3) | DOES_NOT_MEET | No search strategy (databases, time window, query) and no inclusion/exclusion criteria, which reviewers expect from a survey. About 25 PEFT detectors are reviewed |
| Significance | MEETS | Practical recommendations (selection rule, seeds, LN baseline) follow from the evidence |

## C. Submission components (journal_submission_guide)

| Component | Status |
|---|---|
| Conflict-of-interest statement | Missing |
| Funding / Acknowledgements (`\ack`) | Missing |
| Data/code availability | Code URL in the Introduction; no formal statement |
| Ethics note on face datasets (FF++, CDF, DFDC, DF40 licences) | Missing |
| Article type | "Research Article" in the header; the content is a survey. Needs the advisor's/editor's decision |
| Response letter | Covers every reviewer point; numbers match the 10-seed results |

## Verdict

**Not ready to submit yet.** No formatting blockers remain. Content blockers:
1. ~~Survey methodology paragraph~~ — done (Sec. 2.1: arXiv search Jan 2022–Oct 2026, 199 records, criteria I1–I3, 36 included studies; list in `docs/survey_included_studies.csv`). The authors still need to confirm the single-reviewer screening.
2. Add the declarations: conflict of interest, funding/acknowledgements, data availability, dataset-licence/ethics note.
3. Decide the article type with the advisor.

Recommended but not blocking:
4. Benchmark one additive method — adapter and VPT runs in progress on Colab (2026-11-20).
5. GenD seeds 6–9 so every main comparison has n = 10.

## Update 2026-11-20 (evening)

- Fig. 1 is now a single-column vertical overview; the paired-test table and the bias-role table moved to the Appendix; the per-dataset bar chart was removed.
- RQ1–RQ3 replaced by three comparison axes (budget, location, generalization), each with its own section (Secs. 6–8) and a short summary.
- Floats re-ordered so that each table or figure appears on the pages of the section that discusses it. 12 pages, 0 unresolved references.
