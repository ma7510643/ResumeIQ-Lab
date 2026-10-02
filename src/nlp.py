from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src import DATA_DIR

SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "about me"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "experience": ["experience", "work experience", "employment", "internship", "internships"],
    "education": ["education", "academics", "qualifications"],
    "projects": ["projects", "academic projects", "personal projects"],
    "certifications": ["certifications", "certificates", "courses"],
}

DEGREE_PATTERNS = [
    r"b\.?\s*tech",
    r"b\.?\s*e\.?",
    r"m\.?\s*tech",
    r"m\.?\s*c\.?\s*a",
    r"b\.?\s*c\.?\s*a",
    r"b\.?\s*sc",
    r"m\.?\s*sc",
    r"mba",
    r"phd",
    r"bachelor",
    r"master",
    r"diploma",
]

MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"


@lru_cache(maxsize=1)
def load_skill_pack() -> dict:
    return json.loads((DATA_DIR / "skills.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_role_skills() -> dict:
    return json.loads((DATA_DIR / "role_skills.json").read_text(encoding="utf-8"))


def normalize_skill(token: str) -> str:
    pack = load_skill_pack()
    t = token.strip().lower()
    t = pack["normalize"].get(t, t)
    return t


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def mask_identity(text: str) -> str:
    """Drop likely personal identifiers before company scoring."""
    lines = text.splitlines()
    if lines:
        lines[0] = "[CANDIDATE]"
    masked = "\n".join(lines)
    masked = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email]", masked)
    masked = re.sub(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", "[phone]", masked)
    masked = re.sub(r"\b(he/him|she/her|they/them)\b", "", masked, flags=re.I)
    masked = re.sub(r"\b(male|female|mr\.|mrs\.|ms\.)\b", "", masked, flags=re.I)
    return masked


def split_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    buckets: dict[str, list[str]] = {k: [] for k in SECTION_ALIASES}
    buckets["other"] = []
    current = "other"
    for line in lines:
        key = _heading_key(line)
        if key:
            current = key
            continue
        buckets[current].append(line)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def _heading_key(line: str) -> str | None:
    compact = re.sub(r"[^a-zA-Z ]", "", line).strip().lower()
    if not compact or len(compact) > 40:
        return None
    for key, aliases in SECTION_ALIASES.items():
        if compact in aliases:
            return key
    return None


def extract_skills(text: str) -> list[str]:
    pack = load_skill_pack()
    found = []
    lowered = text.lower()
    for skill in sorted(pack["skills"], key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(normalize_skill(skill))
    # de-dupe preserving order
    seen = set()
    unique = []
    for s in found:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def extract_education(text: str) -> list[str]:
    hits = []
    for pat in DEGREE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            start = max(0, m.start() - 10)
            end = min(len(text), m.end() + 60)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip(" -|,")
            hits.append(snippet)
    # unique-ish
    out, seen = [], set()
    for h in hits:
        key = h.lower()
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out[:6]


def extract_experience_years(text: str) -> float:
    years = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text, flags=re.I)
    if years:
        return max(float(y) for y in years)
    # internships / dated ranges as a weak signal
    ranges = re.findall(
        rf"((?:{MONTHS})[a-z]*\.?\s+\d{{4}})\s*[-–to]+\s*((?:{MONTHS})[a-z]*\.?\s+\d{{4}}|present|current)",
        text,
        flags=re.I,
    )
    if ranges:
        return min(4.0, 0.5 * len(ranges) + 0.4)
    if re.search(r"\bintern(ship)?\b", text, flags=re.I):
        return 0.5
    return 0.0


def extract_projects(text: str) -> list[str]:
    section = split_sections(text).get("projects") or ""
    bullets = [re.sub(r"^[\-•\*\d.\)\s]+", "", ln).strip() for ln in section.splitlines()]
    bullets = [b for b in bullets if 12 <= len(b) <= 240]
    if bullets:
        return bullets[:8]
    # fallback: lines with project-ish words
    hits = []
    for ln in text.splitlines():
        if re.search(r"\b(project|developed|built|implemented)\b", ln, flags=re.I) and len(ln.strip()) > 25:
            hits.append(ln.strip())
    return hits[:6]


def parse_resume(text: str, mask: bool = False) -> dict:
    raw = clean_text(text)
    working = mask_identity(raw) if mask else raw
    sections = split_sections(working)
    skills = extract_skills(working)
    return {
        "raw_text": working,
        "sections": sections,
        "skills": skills,
        "education": extract_education(working),
        "experience_years": extract_experience_years(working),
        "projects": extract_projects(working),
        "certifications": [
            ln.strip(" -•")
            for ln in (sections.get("certifications") or "").splitlines()
            if len(ln.strip()) > 4
        ][:8],
    }
