# Formal Project Report

**Title:** *Where Does Document Understanding Break? Counterfactual Error Attribution for Multilingual Indian Document QA*
**Subtitle:** Phase 1 Completion · Confirmatory Study Plan · PARAM Ganga Compute Request
**Version:** 1.0 · 6 November 2026

## Final deliverables

| File | Purpose |
| :--- | :--- |
| `Document_QA_Failure_Analysis_Project_Report.pdf` | Final 26-page PDF for faculty review |
| `Document_QA_Failure_Analysis_Project_Report.docx` | Editable Word source |
| `build_report.py` | Rebuilds the DOCX from the figures and repository evidence |
| `make_figures.py` | Rebuilds the six report figures from pilot outputs |
| `assets/` | Figures used in the report |

## What the report contains

- Cover, executive summary, contents, and list of figures.
- Research questions, objectives, contributions, and scope.
- Related-work positioning against DISCO, Xu et al., DocHallu, IndicVisionBench, and Patram-Bench.
- Architecture-specific counterfactual protocol and identified decompositions.
- Data, question generation, models, perturbation spectrum, metrics, statistics, human validation, and provenance.
- Phase 1 implementation status and pilot evidence.
- Confirmatory study matrix, 250-hour compute estimate, PARAM Ganga specification, timeline, milestones, and risk register.
- Faculty support request and endorsement/signature page.
- Publication plan, ethics/data governance, references, and appendices.

## Rebuild

```bash
# Figures (system Python with matplotlib)
python reports/make_figures.py

# DOCX (bundled python-docx runtime)
"C:\Users\SIDDHARTHA\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" reports/build_report.py

# PDF (LibreOffice Kit)
"C:\Users\SIDDHARTHA\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe" \
  "C:\Users\SIDDHARTHA\AppData\Local\Programs\DeepSeek Harness\resources\app.asar.unpacked\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js" \
  convert --input reports/Document_QA_Failure_Analysis_Project_Report.docx \
  --output reports/Document_QA_Failure_Analysis_Project_Report.pdf
```

## Evidence and caveats

- Phase 1 status reflects the repository at the time of preparation: 31 incremental commits, 58 passing tests, green GitHub Actions on Python 3.10 and 3.11.
- Pilot figures are from `results/local_pilot/`; they are instrumentation evidence, not confirmatory results.
- The report states explicitly that the confirmatory study, including the 7B FP16 scale check, requires PARAM Ganga access.
- Patram-Bench access is described as a request that will be submitted with faculty endorsement; the public fallback is frozen.
- The PDF was generated from the DOCX with LibreOffice and checked for page count, clipping, figure placement, and the `%PDF-` signature. A missing "Courier" font warning is non-fatal; body text uses Calibri/Cambria metric-compatible fonts.
