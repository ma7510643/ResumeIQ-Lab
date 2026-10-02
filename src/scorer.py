from __future__ import annotations

import re

from src.nlp import extract_skills, load_role_skills
from src.parser import looks_ats_friendly


ACTION_VERBS = {
    "led", "built", "developed", "designed", "implemented", "created", "optimized",
    "improved", "automated", "achieved", "delivered", "increased", "reduced", "launched",
    "managed", "analyzed", "engineered", "deployed", "trained", "collaborated",
}

WEAK_PHRASES = [
    r"responsible for",
    r"helped with",
    r"worked on",
    r"duties included",
    r"familiar with",
    r"knowledge of",
]


def _structure_score(parsed: dict) -> dict:
    sections = parsed.get("sections") or {}
    present = [k for k, v in sections.items() if k != "other" and v.strip()]
    needed = ["education", "experience", "skills"]
    missing = [s for s in needed if s not in present]
    score = 8
    if parsed.get("skills"):
        score += 4
    if parsed.get("education"):
        score += 3
    if parsed.get("projects"):
        score += 3
    if "summary" in present:
        score += 2
    if missing:
        score -= 2 * len(missing)
    return {
        "score": max(0, min(20, score)),
        "present_sections": present,
        "missing_core": missing,
    }


def _skills_score(parsed: dict, target_role: str, job_description: str) -> dict:
    resume_skills = set(parsed.get("skills") or [])
    role_map = load_role_skills()
    role_skills = set(role_map.get(target_role, []))
    jd_skills = set(extract_skills(job_description)) if job_description else set()
    needed = role_skills | jd_skills
    if not needed:
        needed = role_skills or resume_skills
    matched = sorted(resume_skills & needed) if needed else sorted(resume_skills)
    missing = sorted(needed - resume_skills) if needed else []
    ratio = (len(matched) / len(needed)) if needed else 0.4
    extra = min(0.15, 0.02 * max(0, len(resume_skills) - 6))
    points = min(30.0, 30.0 * min(1.0, ratio + extra))
    return {
        "score": round(points, 1),
        "matched": matched,
        "missing": missing,
        "resume_skills": sorted(resume_skills),
        "needed": sorted(needed),
    }


def _impact_score(text: str) -> dict:
    bullets = [ln.strip() for ln in text.splitlines() if re.match(r"^[\-•*]", ln.strip()) or len(ln.strip()) > 40]
    if not bullets:
        bullets = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 50][:20]
    action_hits = 0
    metric_hits = 0
    weak_hits = 0
    for b in bullets:
        first = re.split(r"\s+", b.lstrip("-•* ").lower())[0] if b else ""
        if first in ACTION_VERBS:
            action_hits += 1
        if re.search(r"\d+%|\d+\+|\$\d+|\d+\s*(users|clients|projects|hours|days)", b, flags=re.I):
            metric_hits += 1
        if any(re.search(p, b, flags=re.I) for p in WEAK_PHRASES):
            weak_hits += 1
    n = max(1, len(bullets))
    score = 20 * (0.55 * (action_hits / n) + 0.45 * min(1.0, metric_hits / max(3, n * 0.3)))
    score -= min(6, weak_hits * 1.5)
    return {
        "score": round(max(0.0, min(20.0, score)), 1),
        "action_hits": action_hits,
        "metric_hits": metric_hits,
        "weak_hits": weak_hits,
        "bullet_count": len(bullets),
    }


def _length_score(text: str) -> dict:
    words = len(re.findall(r"\b\w+\b", text))
    if 350 <= words <= 800:
        score = 10
        note = "Length looks appropriate for a 1-page professional resume."
    elif 220 <= words < 350:
        score = 7
        note = "A bit short — add 2–3 stronger project or internship bullets."
    elif 800 < words <= 1100:
        score = 7
        note = "Slightly long — trim older or less relevant bullets."
    elif words < 220:
        score = 4
        note = "Too short; add education, skills and quantified experience."
    else:
        score = 3
        note = "Too long for most campus / early-career ATS screens."
    return {"score": score, "word_count": words, "note": note}


def score_resume(parsed: dict, target_role: str, job_description: str, filename: str) -> dict:
    text = parsed.get("raw_text") or ""
    structure = _structure_score(parsed)
    skills = _skills_score(parsed, target_role, job_description)
    impact = _impact_score(text)
    ats = looks_ats_friendly(text, filename)
    length = _length_score(text)
    total = (
        structure["score"]
        + skills["score"]
        + impact["score"]
        + ats["score"]
        + length["score"]
    )
    return {
        "total": round(min(100.0, total), 1),
        "breakdown": {
            "Structure": structure["score"],
            "Skills / Keywords": skills["score"],
            "Impact wording": impact["score"],
            "ATS format": ats["score"],
            "Length": length["score"],
        },
        "max": {
            "Structure": 20,
            "Skills / Keywords": 30,
            "Impact wording": 20,
            "ATS format": 20,
            "Length": 10,
        },
        "structure": structure,
        "skills": skills,
        "impact": impact,
        "ats": ats,
        "length": length,
        "parsed": parsed,
        "target_role": target_role,
    }
