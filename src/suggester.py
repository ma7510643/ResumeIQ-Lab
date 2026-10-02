from __future__ import annotations

import os
import re

from src.scorer import ACTION_VERBS, WEAK_PHRASES

REWRITE_MAP = {
    "responsible for": "Owned",
    "helped with": "Supported",
    "worked on": "Delivered",
    "duties included": "Executed",
    "familiar with": "Applied",
    "knowledge of": "Used",
}


def _heuristic_rewrite(bullet: str) -> str:
    text = bullet.strip(" -•*\t")
    lower = text.lower()
    for weak, strong in REWRITE_MAP.items():
        if lower.startswith(weak):
            text = strong + text[len(weak) :]
            break
    else:
        first = text.split(" ", 1)[0].lower() if text else ""
        if first not in ACTION_VERBS and text:
            text = "Developed " + text[0].lower() + text[1:] if text[:1].isupper() else "Developed " + text
    if not re.search(r"\d", text):
        text = text.rstrip(".") + " — quantified impact (add a metric such as %, users, or time saved)."
    if not text.endswith("."):
        text += "."
    return text


def _llm_rewrite(bullets: list[str], role: str) -> list[str] | None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        import urllib.request
        import json

        prompt = (
            "Rewrite these resume bullets for a "
            f"{role} role. Use action verbs and measurable results. "
            "Return the same number of bullets, one per line, no numbering.\n\n"
            + "\n".join(f"- {b}" for b in bullets)
        )
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
            }
        ).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        lines = [re.sub(r"^[\-\*\d.\)\s]+", "", ln).strip() for ln in text.splitlines() if ln.strip()]
        return lines[: len(bullets)] or None
    except Exception:
        return None


def suggest_improvements(report: dict) -> dict:
    text = report["parsed"]["raw_text"]
    role = report.get("target_role") or "target"
    candidates = []
    for ln in text.splitlines():
        clean = ln.strip()
        if len(clean) < 25:
            continue
        if any(re.search(p, clean, flags=re.I) for p in WEAK_PHRASES) or re.match(r"^[\-•*]", clean):
            candidates.append(re.sub(r"^[\-•*\s]+", "", clean))
        if len(candidates) >= 5:
            break
    if not candidates:
        candidates = [ln.strip() for ln in text.splitlines() if 40 < len(ln.strip()) < 180][:4]

    llm = _llm_rewrite(candidates, role)
    rewrites = []
    for i, original in enumerate(candidates):
        improved = llm[i] if llm and i < len(llm) else _heuristic_rewrite(original)
        rewrites.append({"original": original, "improved": improved})

    tips = []
    skills = report.get("skills") or {}
    if skills.get("missing"):
        tips.append(
            "Add these missing keywords naturally in Skills/Projects: "
            + ", ".join(skills["missing"][:8])
        )
    if report.get("impact", {}).get("metric_hits", 0) < 2:
        tips.append("Quantify 2–3 bullets with %, time saved, users, or accuracy.")
    if report.get("structure", {}).get("missing_core"):
        tips.append("Create missing sections: " + ", ".join(report["structure"]["missing_core"]))
    for issue in (report.get("ats") or {}).get("issues", [])[:3]:
        tips.append(issue)
    if report.get("length", {}).get("note"):
        tips.append(report["length"]["note"])

    return {
        "rewrites": rewrites,
        "tips": tips,
        "used_llm": bool(llm),
    }
