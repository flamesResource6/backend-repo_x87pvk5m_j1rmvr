import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Resume Tailoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TailorRequest(BaseModel):
    resume_text: str = Field(..., description="Full resume text provided by the candidate")
    job_description: str = Field(..., description="Full job description text")
    role_title: Optional[str] = Field(None, description="Optional role title to align the headline")


class TailorResponse(BaseModel):
    tailored_resume: str
    matched_keywords: List[str] = []
    missing_but_referenced_keywords: List[str] = []
    ats_tips: List[str] = []


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


# ----------------- Resume Tailoring Logic -----------------

def _tokenize(text: str) -> List[str]:
    import re
    tokens = re.findall(r"[a-zA-Z0-9+#./-]+", text.lower())
    return tokens


def _extract_keywords(text: str) -> List[str]:
    # Very light heuristic keyword extraction from job description
    tokens = _tokenize(text)
    stop = set([
        'and','or','for','with','to','of','the','a','an','in','on','by','as','at','is','are','be','this','that','you','we','our','your','from','will','ability','experience','years','plus','preferred','required','etc','using','use','can'
    ])
    # Keep tokens of length >= 2 and not in stop words
    kws = [t for t in tokens if len(t) >= 2 and t not in stop]
    # Deduplicate but keep order
    seen = set()
    ordered = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered[:200]


def _split_lines(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines()]
    return [l for l in lines if l]


def _detect_sections(lines: List[str]):
    # Basic section detection by common headings
    import re
    headings = {
        'summary': re.compile(r"^(summary|professional summary|profile)$", re.I),
        'experience': re.compile(r"^(experience|work experience|professional experience)$", re.I),
        'skills': re.compile(r"^(skills|technical skills|core skills|key skills)$", re.I),
        'education': re.compile(r"^(education|academics)$", re.I),
        'projects': re.compile(r"^(projects|selected projects)$", re.I),
        'certifications': re.compile(r"^(certifications|licenses)$", re.I),
        'achievements': re.compile(r"^(achievements|awards)$", re.I),
    }
    sections = {}
    current = 'misc'
    sections[current] = []
    for line in lines:
        matched = False
        for key, rx in headings.items():
            if rx.match(line.lower()):
                current = key
                if current not in sections:
                    sections[current] = []
                matched = True
                break
        if not matched:
            sections.setdefault(current, []).append(line)
    return sections


def _match_keywords_against_resume(jd_kws: List[str], resume_text: str):
    resume_tokens = set(_tokenize(resume_text))
    matched = [k for k in jd_kws if k in resume_tokens]
    missing = [k for k in jd_kws if k not in resume_tokens]
    return matched, missing


def _format_bullets(lines: List[str]) -> List[str]:
    # Ensure bullets are hyphen-prefixed for ATS friendliness
    out = []
    for l in lines:
        if not l:
            continue
        if l.startswith(('-', '•', '*')):
            cleaned = l.lstrip('•* ').strip()
            out.append(f"- {cleaned}")
        else:
            out.append(f"- {l}")
    return out


def _limit_to_relevant_bullets(lines: List[str], jd_kws: List[str]) -> List[str]:
    # Keep bullets that contain any JD keyword; if none, keep original bullets
    jd_set = set(jd_kws)
    relevant = []
    for l in lines:
        toks = set(_tokenize(l))
        if toks & jd_set:
            relevant.append(l)
    return relevant if relevant else lines


def _reorder_skills(skills_lines: List[str], jd_kws: List[str]) -> List[str]:
    # Split skills by commas/slashes and order by JD keyword priority
    import re
    skills_text = ' '.join(skills_lines)
    parts = re.split(r",|/|\|\||;|\n", skills_text)
    skills = [p.strip() for p in parts if p.strip()]
    # Stable sort by presence of JD keyword and by index position in jd_kws
    priority = {kw: i for i, kw in enumerate(jd_kws)}
    def score(skill: str):
        toks = _tokenize(skill)
        best = min([priority[t] for t in toks if t in priority], default=10_000)
        matched = any(t in priority for t in toks)
        return (0 if matched else 1, best)
    skills_sorted = sorted(skills, key=score)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for s in skills_sorted:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(s)
    return ordered


def build_tailored_resume(req: TailorRequest) -> TailorResponse:
    jd_kws = _extract_keywords(req.job_description)
    lines = _split_lines(req.resume_text)
    sections = _detect_sections(lines)

    matched, missing = _match_keywords_against_resume(jd_kws, req.resume_text)

    # Headline
    headline = req.role_title.strip() if req.role_title else None

    # Summary: keep user's original sentences, lightly reorder/trim for JD alignment without inventing
    summary_lines = sections.get('summary', [])
    if summary_lines:
        # Keep up to 3-4 strongest lines that include JD keywords, otherwise first 3
        relevant_summary = _limit_to_relevant_bullets(summary_lines, jd_kws)
        summary_bullets = _format_bullets(relevant_summary[:4])
    else:
        summary_bullets = []

    # Skills: reorder to surface JD-relevant skills first (no new skills added)
    skills_lines = sections.get('skills', [])
    skills_ordered = _reorder_skills(skills_lines, jd_kws) if skills_lines else []

    # Experience: filter bullets to emphasize relevant achievements; never add content that wasn't in resume
    exp_lines = sections.get('experience', [])
    exp_relevant = _limit_to_relevant_bullets(exp_lines, jd_kws)
    exp_bullets = _format_bullets(exp_relevant)

    # Education and other sections preserved as-is
    edu_lines = sections.get('education', [])
    proj_lines = sections.get('projects', [])
    cert_lines = sections.get('certifications', [])
    ach_lines = sections.get('achievements', [])

    # Assemble in an ATS-friendly, simple format
    parts: List[str] = []
    if headline:
        parts.append(headline.upper())
        parts.append("")

    parts.append("SUMMARY")
    if summary_bullets:
        parts.extend(summary_bullets)
    else:
        parts.append("- Results-driven professional. (Tailoring kept minimal because no explicit summary was detected in the original resume.)")
    parts.append("")

    if skills_ordered:
        parts.append("SKILLS")
        parts.append(", ".join(skills_ordered))
        parts.append("")

    if exp_bullets:
        parts.append("EXPERIENCE")
        parts.extend(exp_bullets)
        parts.append("")

    if proj_lines:
        parts.append("PROJECTS")
        parts.extend(_format_bullets(proj_lines))
        parts.append("")

    if edu_lines:
        parts.append("EDUCATION")
        parts.extend(_format_bullets(edu_lines))
        parts.append("")

    if cert_lines:
        parts.append("CERTIFICATIONS")
        parts.extend(_format_bullets(cert_lines))
        parts.append("")

    if ach_lines:
        parts.append("ACHIEVEMENTS")
        parts.extend(_format_bullets(ach_lines))
        parts.append("")

    parts.append("Notes: This tailored resume preserves only information present in the original resume. No new skills, titles, employers, dates, or metrics were fabricated.")

    tailored = "\n".join(parts).strip()

    # ATS tips (advisory; safe and non-fabricated)
    ats_tips = [
        "Use clear section headings like SUMMARY, SKILLS, EXPERIENCE, EDUCATION.",
        "Prefer simple bullets (hyphens) and standard fonts; avoid tables and text boxes.",
        "Mirror exact keywords from the job posting when they reflect your real experience.",
        "Quantify results only if your original bullets already include numbers—do not invent metrics.",
        "Keep file name simple, e.g., FirstName_LastName_Role.pdf",
    ]

    return TailorResponse(
        tailored_resume=tailored,
        matched_keywords=matched[:50],
        missing_but_referenced_keywords=missing[:50],
        ats_tips=ats_tips,
    )


@app.post("/api/tailor", response_model=TailorResponse)
def tailor_resume(req: TailorRequest):
    """
    Transform the provided resume into a job-specific, ATS-optimized version aligned with the job description.
    Important: This endpoint will NEVER fabricate information. It only reorganizes and emphasizes content that
    already exists in the original resume. It will not add employers, titles, dates, certifications, or skills
    that aren't present in the original text.
    """
    return build_tailored_resume(req)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
