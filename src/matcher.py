from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp import extract_skills, normalize_skill, parse_resume


DEFAULT_WEIGHTS = {
    "skill": 0.40,
    "semantic": 0.25,
    "experience": 0.20,
    "education": 0.10,
    "extras": 0.05,
}


def _split_csv(value: str) -> list[str]:
    items = [normalize_skill(p.strip().lower()) for p in re.split(r"[,;\n]", value or "") if p.strip()]
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _semantic_similarity(resume_text: str, job_text: str) -> float:
    model = _try_sbert()
    if model is not None:
        embeddings = model.encode([resume_text[:4000], job_text[:4000]])
        sim = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-9
        ))
        return max(0.0, min(1.0, (sim + 1) / 2 if sim < 0 else sim))
    docs = [resume_text or "empty", job_text or "empty"]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000).fit_transform(docs)
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])


@lru_cache(maxsize=1)
def _try_sbert():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _experience_fit(years: float, minimum: float) -> float:
    if minimum <= 0:
        return 1.0 if years >= 0 else 0.6
    if years >= minimum:
        return 1.0
    if years <= 0:
        return 0.25
    return max(0.3, years / minimum)


def _education_fit(education_items: list[str], required: str) -> float:
    if not required:
        return 1.0
    blob = " ".join(education_items).lower()
    req = required.lower()
    aliases = {
        "b.tech": ["b.tech", "btech", "b.e", "bachelor of technology", "bachelor"],
        "m.tech": ["m.tech", "mtech", "master of technology", "m.e"],
        "bca": ["bca", "bachelor of computer"],
        "mca": ["mca"],
        "mba": ["mba"],
        "any": [],
    }
    keys = aliases.get(req, [req])
    if not keys:
        return 1.0
    return 1.0 if any(k in blob for k in keys) else 0.35


def _extras_score(parsed: dict) -> float:
    score = 0.0
    if parsed.get("projects"):
        score += 0.6
    if parsed.get("certifications"):
        score += 0.4
    return min(1.0, score)


def job_text_blob(job: dict) -> str:
    return " ".join(
        [
            job.get("title") or "",
            job.get("must_have") or "",
            job.get("preferred") or "",
            job.get("education") or "",
            job.get("location") or "",
        ]
    )


def score_candidate(resume_text: str, filename: str, job: dict) -> dict:
    parsed = parse_resume(resume_text, mask=True)
    weights = {**DEFAULT_WEIGHTS, **(job.get("weights") or {})}
    must = _split_csv(job.get("must_have"))
    preferred = _split_csv(job.get("preferred"))
    resume_skills = set(parsed.get("skills") or [])
    needed = set(must) | set(preferred) | set(extract_skills(job_text_blob(job)))
    matched_must = [s for s in must if s in resume_skills]
    missing_must = [s for s in must if s not in resume_skills]
    matched_pref = [s for s in preferred if s in resume_skills]
    skill_den = max(1, len(must) + max(1, len(preferred)) * 0.5)
    skill_num = len(matched_must) + 0.5 * len(matched_pref)
    skill_match = min(1.0, skill_num / skill_den)

    semantic = _semantic_similarity(parsed["raw_text"], job_text_blob(job))
    exp_fit = _experience_fit(parsed.get("experience_years") or 0.0, float(job.get("min_experience") or 0))
    edu_fit = _education_fit(parsed.get("education") or [], job.get("education") or "")
    extras = _extras_score(parsed)

    match = 100.0 * (
        weights["skill"] * skill_match
        + weights["semantic"] * semantic
        + weights["experience"] * exp_fit
        + weights["education"] * edu_fit
        + weights["extras"] * extras
    )
    display_name = f"Candidate {abs(hash(filename)) % 900 + 100}"
    return {
        "filename": filename,
        "display_name": display_name,
        "match_score": round(match, 1),
        "parsed": parsed,
        "explanation": {
            "matched_must": matched_must,
            "missing_must": missing_must,
            "matched_preferred": matched_pref,
            "skill_match": round(skill_match * 100, 1),
            "semantic": round(semantic * 100, 1),
            "experience_fit": round(exp_fit * 100, 1),
            "education_fit": round(edu_fit * 100, 1),
            "extras": round(extras * 100, 1),
            "experience_years": parsed.get("experience_years"),
            "education": parsed.get("education"),
            "skills_found": sorted(resume_skills),
        },
    }
