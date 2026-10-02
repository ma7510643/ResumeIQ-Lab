<<<<<<< HEAD
# ResumeIQ-Lab
ResumeIQ Lab is an AI-powered resume analysis tool built with Streamlit. It helps users evaluate resumes, identify relevant skills, and compare a resume against job requirements through an easy-to-use web interface. This repository contains project source code and setup instructions. Clone repositories, and run Streamlit app locally to get started.
=======
# ResumeIQ

AI-powered resume analyzer, ATS-friendly resume builder, and explainable candidate ranking — built for the AI Lab blueprint (Student + Company modules).

## Screens (from the blueprint)

- Home / login — choose **Student** or **Company**
- Student: Upload & Analyze, Job Match, Resume Builder, My Reports
- Company: Job Setup, Bulk Upload CVs, Ranking with explanations, Shortlist & Export (CSV/PDF)

## Scoring (blueprint)

**Student resume score (100)** = Structure 20 + Skills/keywords 30 + Impact wording 20 + ATS format 20 + Length 10

**Company match** = 0.40 skill + 0.25 semantic (TF-IDF cosine; SBERT if installed) + 0.20 experience + 0.10 education + 0.05 extras  
Weights are editable on the company Job Setup screen. Names / gender / photos are masked before scoring.

## Run locally

```bash
cd C:\Users\HP\ResumeIQ-Lab
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Demo logins:

- Student: `student` / `demo123`
- Company: `company` / `demo123`

Optional live bullet rewrites: set environment variable `GEMINI_API_KEY`. Without it, ResumeIQ uses a rule-based rewriter.

Optional semantic upgrade:

```bash
pip install sentence-transformers
```

The first run will download `all-MiniLM-L6-v2`. If that package is missing, ranking still works with TF-IDF.

## Project layout

- `app.py` — Streamlit UI
- `src/parser.py` — PDF/DOCX text extraction + ATS checks
- `src/nlp.py` — sections, skills, education, experience years
- `src/scorer.py` — student scoring rubric
- `src/matcher.py` — hybrid company ranking
- `src/classifier.py` — Naive Bayes role suggestions
- `src/suggester.py` — LLM / heuristic rewrites
- `src/builder.py` — PDF/DOCX resume export
- `src/db.py` — SQLite users, analyses, jobs, candidates
- `data/` — skill dictionary and role taxonomy
