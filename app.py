from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.builder import build_resume_docx, build_resume_pdf
from src.classifier import suggest_roles
from src.db import (
    authenticate,
    create_user,
    get_analysis,
    init_db,
    list_analyses,
    list_candidates,
    list_jobs,
    save_analysis,
    save_job,
    set_shortlisted,
    replace_candidates,
)
from src.matcher import DEFAULT_WEIGHTS, score_candidate
from src.nlp import load_role_skills, parse_resume
from src.parser import extract_text
from src.scorer import score_resume
from src.suggester import suggest_improvements
from src import UPLOAD_DIR

st.set_page_config(page_title="ResumeIQ", page_icon="🧠", layout="wide")
init_db()

ROLES = list(load_role_skills().keys())

CSS = """
<style>
.stApp { background: #F6F8FB; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.iq-hero { background: linear-gradient(135deg, #0E7C7B 0%, #134E4A 60%, #1B2430 100%);
  color: white; padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1rem; }
.iq-hero h1 { margin: 0; font-size: 1.7rem; }
.iq-hero p { margin: 0.35rem 0 0; opacity: 0.9; }
.iq-card { background: white; border: 1px solid #E5EAF1; border-radius: 14px;
  padding: 1.1rem 1.2rem; box-shadow: 0 8px 24px rgba(27,36,48,0.04); }
.iq-score { font-size: 3.2rem; font-weight: 700; color: #0E7C7B; line-height: 1; }
.iq-muted { color: #5B6777; font-size: 0.92rem; }
.iq-pill { display: inline-block; background: #E6F4F3; color: #0E7C7B;
  border-radius: 999px; padding: 0.15rem 0.7rem; margin: 0.15rem; font-size: 0.85rem; }
.iq-pill.miss { background: #FDECEC; color: #B42318; }
[data-testid="stSidebar"] { background: #102A2A; }
[data-testid="stSidebar"] * { color: #E8F3F2 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _save_upload(file, folder: str) -> Path:
    dest_dir = UPLOAD_DIR / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.name
    dest.write_bytes(file.getbuffer())
    return dest


def login_view() -> None:
    st.markdown(
        """<div class="iq-hero">
        <h1>ResumeIQ</h1>
        <p>AI-powered resume analyzer, builder and smart candidate screening</p>
        </div>""",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### How it works")
        st.write(
            """
            **Students** upload a resume, get a score out of 100, a skill-gap report,
            ATS checks and rewrite suggestions — then generate an ATS-friendly PDF.

            **Companies** set job conditions and weights, bulk-upload CVs, and get an
            explainable ranked shortlist. Names, gender and photos are ignored while scoring.
            """
        )
        st.info("Demo accounts — student / demo123  ·  company / demo123")
    with right:
        tab_login, tab_reg = st.tabs(["Login", "Create account"])
        with tab_login:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Enter ResumeIQ", type="primary", use_container_width=True):
                user = authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.nav = "upload" if user["role"] == "student" else "job"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with tab_reg:
            r_user = st.text_input("New username", key="ru")
            r_pass = st.text_input("Password", type="password", key="rp")
            r_role = st.selectbox("I am a", ["student", "company"])
            if st.button("Create account", use_container_width=True):
                if len(r_user.strip()) < 3 or len(r_pass) < 4:
                    st.warning("Use at least 3 characters for username and 4 for password.")
                else:
                    ok, msg = create_user(r_user, r_pass, r_role)
                    st.success(msg) if ok else st.error(msg)


def student_sidebar() -> str:
    st.sidebar.markdown("### ResumeIQ | Student")
    st.sidebar.caption(st.session_state.user["username"])
    choice = st.sidebar.radio(
        "Navigate",
        ["Upload", "Job Match", "Builder", "My Reports"],
        index=["Upload", "Job Match", "Builder", "My Reports"].index(
            {"upload": "Upload", "match": "Job Match", "builder": "Builder", "reports": "My Reports"}.get(
                st.session_state.get("nav"), "Upload"
            )
        ),
        label_visibility="collapsed",
    )
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()
    return {"Upload": "upload", "Job Match": "match", "Builder": "builder", "My Reports": "reports"}[choice]


def company_sidebar() -> str:
    st.sidebar.markdown("### ResumeIQ | Company")
    st.sidebar.caption(st.session_state.user["username"])
    labels = ["Job Setup", "Upload CVs", "Ranking", "Shortlist"]
    keymap = {"job": "Job Setup", "cvs": "Upload CVs", "rank": "Ranking", "short": "Shortlist"}
    current = keymap.get(st.session_state.get("nav"), "Job Setup")
    choice = st.sidebar.radio("Navigate", labels, index=labels.index(current), label_visibility="collapsed")
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()
    return {"Job Setup": "job", "Upload CVs": "cvs", "Ranking": "rank", "Shortlist": "short"}[choice]


def score_gauge(total: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=total,
            number={"suffix": " / 100", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0E7C7B"},
                "steps": [
                    {"range": [0, 50], "color": "#FDECEC"},
                    {"range": [50, 75], "color": "#FEF3C7"},
                    {"range": [75, 100], "color": "#D1FAE5"},
                ],
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def breakdown_chart(report: dict) -> go.Figure:
    labels = list(report["breakdown"].keys())
    values = list(report["breakdown"].values())
    maxima = [report["max"][k] for k in labels]
    fig = go.Figure()
    fig.add_bar(name="Score", x=labels, y=values, marker_color="#0E7C7B")
    fig.add_bar(name="Max", x=labels, y=maxima, marker_color="#D5DEE8")
    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 32]),
    )
    return fig


def render_report(report: dict, suggestions: dict) -> None:
    c1, c2 = st.columns([0.9, 1.2])
    with c1:
        st.plotly_chart(score_gauge(report["total"]), use_container_width=True)
        st.caption("Resume Score")
    with c2:
        st.plotly_chart(breakdown_chart(report), use_container_width=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Matched skills**")
        matched = report["skills"].get("matched") or []
        if matched:
            st.markdown(" ".join(f'<span class="iq-pill">{x}</span>' for x in matched), unsafe_allow_html=True)
        else:
            st.write("No overlap yet with the target role keywords.")
        st.markdown("**Extracted skills**")
        st.write(", ".join(report["skills"].get("resume_skills") or []) or "—")
        if report["parsed"].get("education"):
            st.markdown("**Education**")
            for e in report["parsed"]["education"]:
                st.write(f"- {e}")
        if report["parsed"].get("projects"):
            st.markdown("**Projects**")
            for p in report["parsed"]["projects"][:5]:
                st.write(f"- {p}")
    with s2:
        st.markdown("**Skill-gap (add these)**")
        missing = report["skills"].get("missing") or []
        if missing:
            st.markdown(" ".join(f'<span class="iq-pill miss">{x}</span>' for x in missing), unsafe_allow_html=True)
        else:
            st.success("No major skill gaps for this role.")
        st.markdown("**ATS notes**")
        issues = report["ats"].get("issues") or ["No major ATS issues detected."]
        for issue in issues:
            st.write(f"- {issue}")
        st.write(report["length"]["note"])

    st.markdown("#### AI suggestions")
    if suggestions.get("used_llm"):
        st.caption("Rewrites generated with a live LLM API.")
    else:
        st.caption("Rule-based rewrites (add GEMINI_API_KEY for live LLM suggestions).")
    for tip in suggestions.get("tips") or []:
        st.write(f"• {tip}")
    for item in suggestions.get("rewrites") or []:
        with st.expander(item["original"][:90] + ("…" if len(item["original"]) > 90 else "")):
            st.write("**Before**")
            st.write(item["original"])
            st.write("**Improved**")
            st.write(item["improved"])

    roles = suggest_roles(report["parsed"]["raw_text"])
    st.markdown("**Suggested roles (classifier)**")
    st.write(" · ".join(f"{r} ({p:.0%})" for r, p in roles))


def analyze_uploaded(file, target_role: str, jd: str, user_id: int) -> None:
    path = _save_upload(file, "resumes")
    text = extract_text(path)
    parsed = parse_resume(text, mask=False)
    report = score_resume(parsed, target_role, jd, file.name)
    suggestions = suggest_improvements(report)
    report["suggestions"] = suggestions
    save_analysis(user_id, file.name, target_role, jd, report["total"], report)
    st.session_state.last_report = report
    st.success(f"Analyzed **{file.name}** — score {report['total']} / 100")
    render_report(report, suggestions)


def student_upload(match_mode: bool = False) -> None:
    st.markdown(
        f"""<div class="iq-hero"><h1>ResumeIQ | Student Dashboard</h1>
        <p>{"Paste a job description and see keyword + semantic fit." if match_mode else "Upload a resume, pick a target role, and get a score with actionable feedback."}</p></div>""",
        unsafe_allow_html=True,
    )
    file = st.file_uploader("Drag & drop your resume here (PDF / DOCX)", type=["pdf", "docx", "txt"])
    target = st.selectbox("Target role", ROLES, index=ROLES.index("Data Analyst") if "Data Analyst" in ROLES else 0)
    jd = ""
    if match_mode:
        jd = st.text_area("Paste job description (optional but recommended)", height=160)
    if st.button("Analyze", type="primary"):
        if not file:
            st.warning("Please upload a resume first.")
        else:
            try:
                analyze_uploaded(file, target, jd, st.session_state.user["id"])
            except Exception as exc:
                st.error(str(exc))


def student_builder() -> None:
    st.markdown(
        """<div class="iq-hero"><h1>ATS-friendly Resume Builder</h1>
        <p>Fill the form, pick a template, download PDF or DOCX.</p></div>""",
        unsafe_allow_html=True,
    )
    last = st.session_state.get("last_report") or {}
    parsed = (last.get("parsed") or {}) if last else {}
    c1, c2 = st.columns(2)
    with c1:
        full_name = st.text_input("Full name", value="Maksud Ansari")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        location = st.text_input("Location", value="India")
        linkedin = st.text_input("LinkedIn / GitHub")
        template = st.selectbox("Template", ["Classic ATS", "Compact ATS"])
    with c2:
        default_skills = ", ".join((parsed.get("skills") or [])[:12])
        skills = st.text_area("Skills (comma or line separated)", value=default_skills, height=90)
        summary = st.text_area(
            "Professional summary",
            value="B.Tech CSE (AI/ML) student seeking a role where I can apply Python, NLP and data skills.",
            height=90,
        )
    education = st.text_area(
        "Education (one per line)",
        value="B.Tech Computer Science Engineering (AI/ML) | 5th Semester",
        height=80,
    )
    experience = st.text_area("Experience / internships (one bullet per line)", height=100)
    projects = st.text_area(
        "Projects (one bullet per line)",
        value="\n".join(parsed.get("projects") or []) or "ResumeIQ — AI resume analyzer and ranking platform.",
        height=100,
    )
    certifications = st.text_area("Certifications", height=70)
    profile = {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "location": location,
        "linkedin": linkedin,
        "skills": skills,
        "summary": summary,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
        "template": template,
    }
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("Download PDF", data=build_resume_pdf(profile), file_name="ResumeIQ_resume.pdf", mime="application/pdf")
    with b2:
        st.download_button(
            "Download DOCX",
            data=build_resume_docx(profile),
            file_name="ResumeIQ_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def student_reports() -> None:
    st.markdown(
        """<div class="iq-hero"><h1>My Reports</h1>
        <p>Previous analyses stored in SQLite.</p></div>""",
        unsafe_allow_html=True,
    )
    rows = list_analyses(st.session_state.user["id"])
    if not rows:
        st.info("No reports yet. Analyze a resume from Upload.")
        return
    df = pd.DataFrame(
        [{"ID": r["id"], "File": r["filename"], "Role": r["target_role"], "Score": r["score"], "When": r["created_at"]} for r in rows]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    chosen = st.selectbox("Open report", [r["id"] for r in rows], format_func=lambda i: f"#{i}")
    record = get_analysis(chosen, st.session_state.user["id"])
    if record:
        render_report(record["report"], record["report"].get("suggestions") or {})


def company_job_setup() -> None:
    st.markdown(
        """<div class="iq-hero"><h1>ResumeIQ | Company Dashboard</h1>
        <p>Create a job post with custom conditions and importance weights.</p></div>""",
        unsafe_allow_html=True,
    )
    title = st.text_input("Role / job title", value="Python Developer")
    must = st.text_input("Must-have skills (comma separated)", value="Python, SQL, ML")
    preferred = st.text_input("Preferred skills", value="Django, Docker, AWS")
    min_exp = st.number_input("Minimum experience (years)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
    education = st.selectbox("Education", ["Any", "B.Tech", "M.Tech", "BCA", "MCA", "MBA"])
    location = st.text_input("Location (optional)", value="Remote / India")
    st.markdown("**Weights** (must sum conceptually; they are normalized on save)")
    w1, w2, w3, w4, w5 = st.columns(5)
    skill = w1.slider("Skills", 0.0, 1.0, DEFAULT_WEIGHTS["skill"])
    semantic = w2.slider("Semantic", 0.0, 1.0, DEFAULT_WEIGHTS["semantic"])
    experience = w3.slider("Experience", 0.0, 1.0, DEFAULT_WEIGHTS["experience"])
    edu_w = w4.slider("Education", 0.0, 1.0, DEFAULT_WEIGHTS["education"])
    extras = w5.slider("Extras", 0.0, 1.0, DEFAULT_WEIGHTS["extras"])
    total_w = skill + semantic + experience + edu_w + extras or 1.0
    weights = {
        "skill": skill / total_w,
        "semantic": semantic / total_w,
        "experience": experience / total_w,
        "education": edu_w / total_w,
        "extras": extras / total_w,
    }
    if st.button("Save job criteria", type="primary"):
        job_id = save_job(
            st.session_state.user["id"],
            {
                "title": title,
                "must_have": must,
                "preferred": preferred,
                "min_experience": min_exp,
                "education": education,
                "location": location,
                "weights": weights,
            },
        )
        st.session_state.active_job_id = job_id
        st.success(f"Saved job #{job_id} — {title}")
        st.session_state.nav = "cvs"
        st.rerun()
    jobs = list_jobs(st.session_state.user["id"])
    if jobs:
        st.markdown("**Existing jobs**")
        st.dataframe(
            pd.DataFrame(
                [{"ID": j["id"], "Title": j["title"], "Must-have": j["must_have"], "Min exp": j["min_experience"]} for j in jobs]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _active_job():
    jobs = list_jobs(st.session_state.user["id"])
    if not jobs:
        st.warning("Create a job first in Job Setup.")
        return None
    ids = [j["id"] for j in jobs]
    default = st.session_state.get("active_job_id", ids[0])
    idx = ids.index(default) if default in ids else 0
    selected = st.selectbox("Active job", ids, index=idx, format_func=lambda i: next(j["title"] + f" (#{j['id']})" for j in jobs if j["id"] == i))
    st.session_state.active_job_id = selected
    return next(j for j in jobs if j["id"] == selected)


def company_upload() -> None:
    st.markdown(
        """<div class="iq-hero"><h1>Bulk upload CVs</h1>
        <p>Names, gender and photos are ignored while scoring.</p></div>""",
        unsafe_allow_html=True,
    )
    job = _active_job()
    if not job:
        return
    files = st.file_uploader("Upload many resumes at once", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Parse & rank", type="primary") and files:
        ranked = []
        errors = []
        for f in files:
            try:
                path = _save_upload(f, "company")
                text = extract_text(path)
                ranked.append(score_candidate(text, f.name, job))
            except Exception as exc:
                errors.append(f"{f.name}: {exc}")
        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        replace_candidates(job["id"], ranked)
        st.session_state.nav = "rank"
        if errors:
            st.warning("Some files failed:\n" + "\n".join(errors))
        st.success(f"Ranked {len(ranked)} candidate(s).")
        st.rerun()
    elif st.button("Parse & rank", type="primary"):
        st.warning("Select at least one resume.")


def _candidate_table(job_id: int, shortlist_only: bool = False) -> pd.DataFrame:
    rows = list_candidates(job_id)
    if shortlist_only:
        rows = [r for r in rows if r["shortlisted"]]
    records = []
    for i, r in enumerate(rows, start=1):
        exp = r["explanation"]
        matched = len(exp.get("matched_must") or [])
        must_n = matched + len(exp.get("missing_must") or [])
        records.append(
            {
                "#": i,
                "id": r["id"],
                "Candidate": r["display_name"],
                "File": r["filename"],
                "Match %": r["match_score"],
                "Skills": f"{matched} / {must_n}" if must_n else "—",
                "Shortlist": bool(r["shortlisted"]),
            }
        )
    return pd.DataFrame(records)


def company_ranking() -> None:
    st.markdown(
        """<div class="iq-hero"><h1>Ranking</h1>
        <p>Hybrid score = skills + semantic similarity + experience + education + extras.</p></div>""",
        unsafe_allow_html=True,
    )
    job = _active_job()
    if not job:
        return
    st.markdown(
        f"""<div class="iq-card"><b>Job criteria</b><br>
        Role: {job['title']}<br>
        Must-have: {job['must_have']}<br>
        Min. experience: {job['min_experience']} yr<br>
        Education: {job['education']}</div>""",
        unsafe_allow_html=True,
    )
    df = _candidate_table(job["id"])
    if df.empty:
        st.info("Upload CVs to see ranking.")
        return
    st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)
    pick = st.selectbox("Explain candidate", df["id"].tolist(), format_func=lambda i: df.loc[df["id"] == i, "Candidate"].iloc[0])
    detail = next(c for c in list_candidates(job["id"]) if c["id"] == pick)
    exp = detail["explanation"]
    a, b = st.columns(2)
    with a:
        st.markdown("**Matched must-have**")
        st.write(", ".join(exp.get("matched_must") or []) or "—")
        st.markdown("**Preferred matched**")
        st.write(", ".join(exp.get("matched_preferred") or []) or "—")
        st.write(f"Experience fit: {exp.get('experience_fit')}% ({exp.get('experience_years')} yrs parsed)")
    with b:
        st.markdown("**Missing must-have**")
        st.write(", ".join(exp.get("missing_must") or []) or "None")
        st.write(f"Semantic similarity: {exp.get('semantic')}%")
        st.write(f"Education fit: {exp.get('education_fit')}%")
        st.write(f"Projects/certs extras: {exp.get('extras')}%")
    flag = st.checkbox("Shortlist this candidate", value=bool(detail["shortlisted"]))
    if flag != bool(detail["shortlisted"]):
        set_shortlisted(detail["id"], flag)
        st.rerun()


def company_shortlist() -> None:
    st.markdown(
        """<div class="iq-hero"><h1>Shortlist & export</h1>
        <p>Filter, compare and download CSV for the hiring team.</p></div>""",
        unsafe_allow_html=True,
    )
    job = _active_job()
    if not job:
        return
    min_score = st.slider("Minimum match %", 0, 100, 60)
    rows = [r for r in list_candidates(job["id"]) if r["match_score"] >= min_score]
    if not rows:
        st.info("No candidates in this filter.")
        return
    compare_ids = st.multiselect(
        "Side-by-side compare",
        [r["id"] for r in rows],
        format_func=lambda i: next(r["display_name"] for r in rows if r["id"] == i),
        max_selections=3,
    )
    table = []
    for r in rows:
        exp = r["explanation"]
        table.append(
            {
                "Candidate": r["display_name"],
                "File": r["filename"],
                "Match %": r["match_score"],
                "Matched skills": ", ".join(exp.get("matched_must") or []),
                "Missing skills": ", ".join(exp.get("missing_must") or []),
                "Experience yrs": exp.get("experience_years"),
                "Shortlisted": "Yes" if r["shortlisted"] else "No",
            }
        )
    df = pd.DataFrame(table)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if compare_ids:
        cols = st.columns(len(compare_ids))
        lookup = {r["id"]: r for r in rows}
        for col, cid in zip(cols, compare_ids):
            r = lookup[cid]
            with col:
                st.markdown(f"**{r['display_name']}**")
                st.metric("Match", f"{r['match_score']}%")
                st.write("Matched:", ", ".join(r["explanation"].get("matched_must") or []) or "—")
                st.write("Missing:", ", ".join(r["explanation"].get("missing_must") or []) or "—")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Export CSV", data=csv_buf.getvalue(), file_name=f"shortlist_job_{job['id']}.csv", mime="text/csv")
    # simple text "PDF" report via reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf_io = io.BytesIO()
    c = canvas.Canvas(pdf_io, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"ResumeIQ shortlist — {job['title']}")
    y -= 24
    c.setFont("Helvetica", 10)
    for row in table[:28]:
        line = f"{row['Candidate']}  {row['Match %']}%  missing: {row['Missing skills']}"
        c.drawString(40, y, line[:110])
        y -= 16
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    st.download_button("Export PDF", data=pdf_io.getvalue(), file_name=f"shortlist_job_{job['id']}.pdf", mime="application/pdf")


def main() -> None:
    user = st.session_state.get("user")
    if not user:
        login_view()
        return
    if user["role"] == "student":
        nav = student_sidebar()
        st.session_state.nav = nav
        if nav == "upload":
            student_upload(False)
        elif nav == "match":
            student_upload(True)
        elif nav == "builder":
            student_builder()
        else:
            student_reports()
    else:
        nav = company_sidebar()
        st.session_state.nav = nav
        if nav == "job":
            company_job_setup()
        elif nav == "cvs":
            company_upload()
        elif nav == "rank":
            company_ranking()
        else:
            company_shortlist()


if __name__ == "__main__":
    main()
