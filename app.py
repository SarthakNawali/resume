"""
ATS Resume Scorer
=================
Run:     streamlit run ats_app.py
Install: pip install streamlit pdfplumber spacy sentence-transformers scikit-learn joblib
         python -m spacy download en_core_web_sm

Place ats_models/ folder (from training notebook) next to this file.
"""

import re, io, os, datetime, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import pdfplumber
import streamlit as st

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ATS Resume Scorer", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #1c2333 0%, #1a2744 60%, #1c2520 100%);
    border-radius: 16px;
    padding: 32px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
    border: 1px solid #2a3555;
}
.hero-title  { font-size:26px; font-weight:800; color:#f0f4ff; margin:0 0 8px; }
.hero-sub    { font-size:14px; color:#94a3b8; margin:0 0 16px; line-height:1.6; max-width:480px; }
.hero-badge  {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:20px;
    font-size:13px; font-weight:600; border:1.5px solid;
}

/* ── SVG gauge wrapper ── */
.gauge-wrap { text-align:center; }

/* ── section heading ── */
.sec-head {
    font-size:20px; font-weight:700; color:#1e293b;
    margin: 8px 0 18px; display:flex; align-items:center; gap:10px;
}

/* ── section card ── */
.card {
    background:#0f172a;
    border:1px solid #1e293b;
    border-radius:14px;
    padding:20px;
    height:100%;
    color:#e2e8f0;
}
.card-row    { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }
.card-title  { font-size:15px; font-weight:600; color:#cbd5e1; }
.card-score  { font-size:26px; font-weight:800; line-height:1; }
.prog-bg     { background:#1e293b; border-radius:6px; height:6px; margin-bottom:14px; overflow:hidden; }
.prog-fill   { height:100%; border-radius:6px; }
.card-msg    { font-size:13px; color:#64748b; margin-bottom:12px; line-height:1.5; }
.card-tip    {
    display:flex; gap:8px; align-items:flex-start;
    font-size:13px; color:#94a3b8; line-height:1.5;
    padding:9px 12px; background:#162032; border-radius:8px; margin-bottom:6px;
}
.tip-arrow   { color:#6366f1; flex-shrink:0; font-size:14px; }

/* ── ATS mini cards ── */
.mini-card  {
    background:#0f172a; border:1px solid #1e293b;
    border-radius:10px; padding:14px 12px; text-align:center;
}
.mini-label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.mini-score { font-size:22px; font-weight:800; }
.mini-max   { font-size:12px; color:#475569; }

/* ── keyword chips ── */
.chip       { display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; margin:3px 2px; font-weight:500; }
.chip-g     { background:#d1fae5; color:#065f46; }
.chip-r     { background:#fee2e2; color:#991b1b; }
.chip-y     { background:#fef9c3; color:#854d0e; }

/* ── upload / button tweaks ── */
div[data-testid="stFileUploader"] section {
    border: 2px dashed #334155 !important;
    border-radius: 10px !important;
    background: #0f172a !important;
}
div[data-testid="stButton"] > button {
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    width: 100% !important;
}
textarea {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
TFIDF_PKL = os.path.join(BASE, "ats_models", "tfidf_vectorizer.pkl")
RF_PKL    = os.path.join(BASE, "ats_models", "rf_ats_scorer.pkl")

# ─── Skill & resume patterns ──────────────────────────────────────────────────
SKILL_PAT = [
    r'\bpython\b',r'\bmatlab\b',r'\bc\+\+\b',r'\bc/c\+\+\b',r'\bjava\b',
    r'\bsql\b',r'\blabview\b',r'\bspice\b',r'\bpcb\b',r'\brf\b',
    r'\bfpga\b',r'\bvhdl\b',r'\bverilog\b',r'\bawr\b',r'\bansoft\b',
    r'\bsas\b',r'\bspss\b',r'\bexcel\b',r'\bnumpy\b',r'\bpandas\b',
    r'\btensorflow\b',r'\bkeras\b',r'\bscikit\b',r'\bgit\b',r'\bdocker\b',
    r'\blinux\b',r'\baws\b',r'\bazure\b',r'\bgcp\b',r'\bjavascript\b',
    r'\bhtml\b',r'\bcss\b',r'\breact\b',r'\bnode\b',r'\bdjango\b',
    r'\bflask\b',r'\bmachine learning\b',r'\bdeep learning\b',
    r'\bdata analysis\b',r'\bdata science\b',r'\bstatistics\b',
    r'\bagile\b',r'\bscrum\b',r'\bdevops\b',r'\bkubernetes\b',
    r'\brest api\b',r'\bpower bi\b',r'\btableau\b',r'\bspark\b',
    r'\bmongodb\b',r'\bpostgresql\b',r'\bmysql\b',
]
RESUME_PAT = [
    r'\bexperience\b',r'\beducation\b',r'\bskills\b',r'\bsummary\b',
    r'\bobjective\b',r'\bcertification\b',r'\bproject',r'\bemployment\b',
    r'\bqualification\b',r'\bachievement\b',r'\binternship\b',
    r'(bachelor|master|degree|b\.s|m\.s|ph\.d)',
    r'\bgpa\b',r'\buniversity\b',r'\bcollege\b',
]

# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    errs = []
    try:    nlp = spacy.load("en_core_web_sm")
    except: nlp = None; errs.append("spaCy missing — run: python -m spacy download en_core_web_sm")
    try:    sm = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e: sm = None; errs.append(f"SentenceTransformer: {e}")
    tfidf = None
    if os.path.exists(TFIDF_PKL):
        try: tfidf = joblib.load(TFIDF_PKL)
        except Exception as e: errs.append(f"TF-IDF load error: {e}")
    else:
        errs.append("ats_models/tfidf_vectorizer.pkl not found — run training notebook first")
    rf = None
    if os.path.exists(RF_PKL):
        try: rf = joblib.load(RF_PKL)
        except Exception as e: errs.append(f"RF load error: {e}")
    else:
        errs.append("ats_models/rf_ats_scorer.pkl not found — run training notebook first")
    return nlp, sm, tfidf, rf, errs

# ─── PDF validation ────────────────────────────────────────────────────────────
def validate_pdf(f):
    if f.type != "application/pdf":
        return dict(ok=False, msg="Not a PDF file.", text="", tables=False)
    raw = f.read()
    if not raw.startswith(b"%PDF"):
        return dict(ok=False, msg="Invalid PDF (bad header).", text="", tables=False)
    text, tables = "", False
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text()
                if t: text += t + "\n"
                if pg.extract_tables(): tables = True
    except Exception as e:
        return dict(ok=False, msg=f"Cannot parse PDF: {e}", text="", tables=False)
    if len(text.strip()) < 100:
        return dict(ok=False, msg="No readable text found. Scanned PDFs are not supported.", text="", tables=False)
    hits = sum(1 for p in RESUME_PAT if re.search(p, text.lower()))
    if hits < 3:
        return dict(ok=False, msg=f"Doesn't look like a resume ({hits}/3 sections found). Add Experience, Education, and Skills sections.", text=text, tables=tables)
    return dict(ok=True, msg="Valid resume.", text=text, tables=tables)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7F]+", " ", t)).strip().lower()

def get_kw(text, nlp):
    doc = nlp(text[:50000]); kws = set()
    for tok in doc:
        if not tok.is_stop and not tok.is_punct and tok.is_alpha and len(tok.text) > 2 \
                and tok.pos_ in ("NOUN","PROPN","ADJ"):
            kws.add(tok.lemma_.lower())
    for ch in doc.noun_chunks:
        c = ch.text.strip().lower()
        if 2 <= len(c.split()) <= 4: kws.add(c)
    return list(kws)[:80]

def get_skills(t):
    return list({p.replace(r"\b","").replace("\\","").strip()
                 for p in SKILL_PAT if re.search(p, t, re.IGNORECASE)})

def get_years(t):
    for pat in [r"(\d+)\+?\s*years?\s+of\s+experience",
                r"(\d+)\+?\s*years?\s+experience"]:
        m = re.search(pat, t, re.IGNORECASE)
        if m: return float(m.group(1))
    ranges = re.findall(r"(\d{4})\s*(?:to|-)\s*(\d{4}|current|present)", t, re.IGNORECASE)
    cy = datetime.datetime.now().year
    return float(sum(max(0,(cy if e.lower() in("current","present") else int(e))-int(s)) for s,e in ranges))

def score_color(s):
    if s >= 80: return "#22c55e"
    if s >= 60: return "#eab308"
    if s >= 40: return "#f97316"
    return "#ef4444"

# ─── Section scorers ───────────────────────────────────────────────────────────
def score_contact(text):
    checks = {
        "email"   : r"[\w.+-]+@[\w-]+\.\w+",
        "phone"   : r"(\+?\d[\d\s\-().]{7,})",
        "linkedin": r"linkedin\.com",
        "location": r"\b(\d{5}|remote|[A-Z][a-z]+,\s*[A-Z]{2})\b",
    }
    tips = []
    score = 0
    for key, pat in checks.items():
        if re.search(pat, text, re.IGNORECASE):
            score += 25
        else:
            tip_map = {
                "email"   : "Add a professional email address",
                "phone"   : "Include a phone number",
                "linkedin": "Add your LinkedIn profile URL",
                "location": "Add your city/state or mention 'Remote'",
            }
            tips.append(tip_map[key])
    msg = "Good contact information provided" if score >= 75 else \
          "Some contact details are missing" if score >= 50 else \
          "Contact section is incomplete"
    return min(score, 100), msg, tips

def score_summary(text):
    has_heading = bool(re.search(r"\b(summary|objective|profile|about me|professional summary)\b", text, re.IGNORECASE))
    # Count lines after summary heading
    lines = text.split("\n")
    in_sec, body_lines = False, 0
    for line in lines:
        if re.search(r"\b(summary|objective|profile)\b", line, re.IGNORECASE): in_sec = True; continue
        if in_sec:
            stripped = line.strip()
            if stripped and not re.search(r"\b(experience|education|skills|work)\b", stripped, re.IGNORECASE):
                body_lines += 1
            elif body_lines > 0: break

    if not has_heading:
        return 30, "No clear professional summary found", [
            "Add a professional summary at the top highlighting your key strengths",
            "Include 3-4 sentences covering your experience, skills, and career goals",
        ]
    if body_lines < 2:
        return 60, "Summary section found but it's too brief", [
            "Expand your summary to 3-4 sentences",
            "Mention your top skills, years of experience, and career goal",
        ]
    return 90, "Strong professional summary present", [
        "Tailor your summary keywords to match each specific job description",
    ]

def score_experience(text):
    has_section = bool(re.search(r"\b(experience|work history|employment|positions?)\b", text, re.IGNORECASE))
    has_bullets = (text.count("•") + text.count("·") + text.count("-")) > 3
    has_dates   = bool(re.search(r"\d{4}\s*[-–]\s*(\d{4}|present|current)", text, re.IGNORECASE))
    has_numbers = bool(re.search(r"\d+\s*(%|percent|million|\$|k\b|x\b)", text, re.IGNORECASE))

    score = 0; tips = []
    if has_section: score += 35
    else: tips.append("Add a clear 'Work Experience' section heading")
    if has_bullets: score += 25
    else: tips.append("Use bullet points to describe your responsibilities and achievements")
    if has_dates:   score += 25
    else: tips.append("Add start and end dates for each role (e.g. Jan 2020 – Present)")
    if has_numbers: score += 15
    else: tips.append("Quantify achievements with numbers (e.g. 'Increased efficiency by 30%')")

    msg = "Strong experience section with good details"    if score >= 85 else \
          "Good experience section, minor improvements needed" if score >= 65 else \
          "Experience section needs more detail"            if score >= 40 else \
          "Experience section is missing or very weak"
    return min(score, 100), msg, tips

def score_skills(text, res_skills, jd_skills, missing_skills):
    has_section  = bool(re.search(r"\b(skills|technologies|tools|competencies|expertise|proficiencies)\b", text, re.IGNORECASE))
    matched      = [s for s in jd_skills if s in res_skills]
    coverage_pct = len(matched) / max(len(jd_skills), 1)

    score = 0; tips = []
    if has_section: score += 40
    else: tips.append("Add a dedicated 'Skills & Technologies' section")
    score += int(coverage_pct * 60)

    if len(res_skills) < 5:
        tips.append("Add more technical and domain-specific skills")
    if missing_skills:
        tips.append(f"Consider adding in-demand skills: {', '.join(missing_skills[:5])}")
    if not tips:
        tips.append("Keep your skills section updated as you gain new tools")

    msg = "Comprehensive skills section present"          if score >= 80 else \
          "Skills section could be more comprehensive"    if score >= 50 else \
          "Skills section is weak or missing"
    return min(score, 100), msg, tips

def score_education(text):
    has_edu   = bool(re.search(r"\b(education|degree|university|college|bachelor|master|ph\.?d|institute)\b", text, re.IGNORECASE))
    has_gpa   = bool(re.search(r"\bgpa\b", text, re.IGNORECASE))
    has_year  = bool(re.search(r"(20\d{2}|19\d{2})", text))
    has_certs = bool(re.search(r"\b(certified|certification|certificate|coursework|course|bootcamp|aws|pmp)\b", text, re.IGNORECASE))

    score = 0; tips = []
    if has_edu:   score += 55
    else:         tips.append("Add your educational background (degree, institution, year)")
    if has_year:  score += 15
    else:         tips.append("Include your graduation year")
    if has_gpa:   score += 15
    else:         tips.append("Include GPA if it is 3.5 or above")
    if has_certs: score += 15
    else:         tips.append("Include relevant coursework or certifications")

    msg = "Strong education section"                   if score >= 80 else \
          "Education section present"                  if score >= 55 else \
          "Education section is incomplete or missing"
    return min(score, 100), msg, tips

# ─── Master scorer ─────────────────────────────────────────────────────────────
def run_scoring(resume_raw, has_tables, jd_raw, nlp, sm, tfidf, rf):
    rc = clean(resume_raw)
    jc = clean(jd_raw)

    # Keywords
    jd_kw      = get_kw(jc, nlp)
    matched_kw = [k for k in jd_kw if k in rc]
    missing_kw = [k for k in jd_kw if k not in rc]

    # Skills
    jd_skills  = get_skills(jc)
    res_skills = get_skills(rc)
    miss_sk    = [s for s in jd_skills if s not in res_skills]

    # Years
    req_yrs = get_years(jc)
    res_yrs = get_years(rc)

    # TF-IDF similarity (trained model)
    if tfidf:
        try:
            rv = tfidf.transform([rc]); jv = tfidf.transform([jc])
            tfidf_sim = float(cosine_similarity(rv, jv)[0][0])
        except: tfidf_sim = len(matched_kw)/max(len(jd_kw),1)
    else:
        tfidf_sim = len(matched_kw)/max(len(jd_kw),1)

    # Semantic similarity
    re_emb = sm.encode([rc[:3000]], convert_to_numpy=True)
    jd_emb = sm.encode([jc[:3000]], convert_to_numpy=True)
    sem_sim = float(cosine_similarity(re_emb, jd_emb)[0][0])

    # ATS component scores
    kw_score  = round((0.6*tfidf_sim + 0.4*(len(matched_kw)/max(len(jd_kw),1))) * 40, 2)
    sem_score = round(sem_sim * 30, 2)
    sk_score  = round(len([s for s in jd_skills if s in res_skills]) / max(len(jd_skills),1) * 10, 2)
    if req_yrs > 0:
        exp_score = round(min(res_yrs/req_yrs, 1.0) * 15, 2)
    else:
        exp_score = 15.0 if res_yrs > 0 else 7.5

    wc         = len(rc.split())
    spec_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,;:\-/()]", resume_raw)) / max(len(resume_raw),1)
    fmt = 5.0; fmt_issues = []
    if has_tables:     fmt -= 2; fmt_issues.append("Replace tables with bullet lists for better ATS parsing")
    if spec_ratio>0.05: fmt -= 1; fmt_issues.append(f"Reduce special characters (current: {spec_ratio:.1%})")
    if wc < 300:       fmt -= 2; fmt_issues.append(f"Resume too short ({wc} words) — aim for 400+")
    fmt_score = max(0.0, fmt)

    rule_total = min(kw_score+sem_score+sk_score+exp_score+fmt_score, 100.0)

    # RF prediction (trained on corpus)
    rf_score = None; used_rf = False
    if rf:
        try:
            feats = {
                "keyword_score": kw_score, "semantic_score": sem_score,
                "skills_score": sk_score, "experience_score": exp_score,
                "formatting_score": fmt_score, "resume_years": res_yrs,
                "word_count": wc, "cosine_sim": sem_sim,
                "matched_keywords": len(matched_kw),
                "matched_skills": len([s for s in jd_skills if s in res_skills]),
            }
            X = np.array([[feats.get(f,0) for f in rf["features"]]])
            rf_score = float(rf["model"].predict(X)[0])
            rf_score = round(min(max(rf_score,0),100), 1)
            used_rf  = True
        except: pass

    final = round(0.6*rf_score + 0.4*rule_total, 1) if used_rf else round(rule_total, 1)
    final = min(final, 100.0)

    # Grade
    if   final >= 85: grade, gc = "Excellent",     "#22c55e"
    elif final >= 70: grade, gc = "Good Progress",  "#eab308"
    elif final >= 50: grade, gc = "Average",         "#f97316"
    else:             grade, gc = "Needs Work",      "#ef4444"

    hero_msgs = {
        "Excellent":    "Outstanding! Your resume is highly optimised and ready to impress recruiters.",
        "Good Progress":"Good foundation! With some improvements, your resume can stand out more to recruiters.",
        "Average":      "Your resume shows potential but needs several improvements to pass ATS filters.",
        "Needs Work":   "Your resume needs substantial improvements to be competitive in ATS screening.",
    }

    # Section scores
    contact_sc,  contact_msg,  contact_tips  = score_contact(resume_raw)
    summary_sc,  summary_msg,  summary_tips  = score_summary(resume_raw)
    exp_sc,      exp_msg,      exp_tips      = score_experience(resume_raw)
    skills_sc,   skills_msg,   skills_tips   = score_skills(rc, res_skills, jd_skills, miss_sk)
    edu_sc,      edu_msg,      edu_tips      = score_education(resume_raw)

    overall_section_score = round((contact_sc+summary_sc+exp_sc+skills_sc+edu_sc)/5)

    return dict(
        final=final, rule_total=round(rule_total,1),
        rf_score=rf_score, used_rf=used_rf,
        grade=grade, grade_color=gc,
        hero_msg=hero_msgs[grade],
        overall_section=overall_section_score,
        sections=dict(
            contact    = dict(score=contact_sc,  title="Contact Information",   msg=contact_msg,  tips=contact_tips),
            summary    = dict(score=summary_sc,  title="Professional Summary",  msg=summary_msg,  tips=summary_tips),
            experience = dict(score=exp_sc,      title="Work Experience",        msg=exp_msg,       tips=exp_tips),
            skills     = dict(score=skills_sc,   title="Skills & Technologies", msg=skills_msg,   tips=skills_tips),
            education  = dict(score=edu_sc,      title="Education",              msg=edu_msg,       tips=edu_tips),
        ),
        kw_score=kw_score, sem_score=sem_score, sk_score=sk_score,
        exp_score=exp_score, fmt_score=fmt_score,
        matched_kw=matched_kw, missing_kw=missing_kw[:20],
        jd_skills=jd_skills, res_skills=res_skills, miss_skills=miss_sk,
        req_yrs=req_yrs, res_yrs=res_yrs,
        wc=wc, fmt_issues=fmt_issues,
        tfidf_sim=round(tfidf_sim,4), sem_sim=round(sem_sim,4),
    )

# ─── SVG circular gauge ────────────────────────────────────────────────────────
def svg_gauge(score, color="#eab308", size=160):
    r = 58; cx = cy = 80
    start, span = 135, 270
    filled = span * (score / 100)
    def arc(sd, sw):
        sr = math.radians(sd); er = math.radians(sd+sw)
        x1,y1 = cx+r*math.cos(sr), cy+r*math.sin(sr)
        x2,y2 = cx+r*math.cos(er), cy+r*math.sin(er)
        lg = 1 if sw > 180 else 0
        return f"M{x1:.2f},{y1:.2f} A{r},{r} 0 {lg},1 {x2:.2f},{y2:.2f}"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 160 160">
      <path d="{arc(start,span)}"  fill="none" stroke="#2a3555" stroke-width="12" stroke-linecap="round"/>
      <path d="{arc(start,max(filled,0.1))}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"/>
      <text x="80" y="76" text-anchor="middle" font-family="Inter,sans-serif"
            font-size="30" font-weight="800" fill="{color}">{score:.0f}</text>
      <text x="80" y="94" text-anchor="middle" font-family="Inter,sans-serif"
            font-size="11" fill="#64748b">out of 100</text>
    </svg>"""

# ─── Section card renderer ──────────────────────────────────────────────────────
def render_card(data):
    s   = data["score"]
    clr = score_color(s)
    tips_html = "".join(
        f'<div class="card-tip"><span class="tip-arrow">›</span><span>{tip}</span></div>'
        for tip in data["tips"][:3]
    )
    return f"""
    <div class="card">
        <div class="card-row">
            <span class="card-title">{data['title']}</span>
            <span class="card-score" style="color:{clr};">{s}</span>
        </div>
        <div class="prog-bg">
            <div class="prog-fill" style="width:{s}%; background:{clr};"></div>
        </div>
        <div class="card-msg">{data['msg']}</div>
        {tips_html}
    </div>"""

def chips(items, cls):
    if not items: return "<i style='color:#64748b;font-size:13px;'>None</i>"
    return "".join(f'<span class="chip {cls}">{i}</span>' for i in items)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("📄 ATS Resume Scorer")
st.caption("Upload your resume PDF and paste a job description to get a detailed ATS score.")
st.divider()

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Resume PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed",
                                 help="Text-based PDF only. Scanned images not supported.")
    if uploaded:
        st.success(f"📎 **{uploaded.name}** · {uploaded.size/1024:.1f} KB")

with c2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste JD", height=200, label_visibility="collapsed",
        placeholder="Paste the full job description here...\n\nInclude responsibilities, requirements, and required skills.")

st.markdown("")
_, mid, _ = st.columns([2, 1, 2])
with mid:
    go = st.button("⚡ Analyse Resume", type="primary", use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if go:
    if not uploaded:
        st.error("Please upload a resume PDF.")
        st.stop()
    if not jd_text or len(jd_text.strip()) < 50:
        st.error("Please paste a job description (minimum 50 characters).")
        st.stop()

    with st.spinner("Validating PDF…"):
        val = validate_pdf(uploaded)
    if not val["ok"]:
        st.error(f"**File rejected:** {val['msg']}")
        with st.expander("What makes a valid resume PDF?"):
            st.markdown("- Real PDF file (not a renamed image or Word doc)\n- Selectable text (not a scanned image)\n- Contains resume sections: Experience, Education, Skills, etc.")
        st.stop()

    ph = st.info("Loading models… (first run ~30 seconds)")
    nlp, sm, tfidf, rf, errs = load_models()
    ph.empty()
    for e in errs: st.warning(f"⚠ {e}")
    if not nlp or not sm:
        st.error("Core NLP models failed to load.")
        st.stop()

    with st.spinner("Computing your ATS score…"):
        R = run_scoring(val["text"], val["tables"], jd_text, nlp, sm, tfidf, rf)

    # ─────────────────────────────────────────────────────────────────────────
    # HERO BANNER — Overall Resume Score
    # ─────────────────────────────────────────────────────────────────────────
    gc    = R["grade_color"]
    gauge = svg_gauge(R["final"], gc, 160)
    st.markdown(f"""
    <div class="hero">
        <div>
            <div class="hero-title">Overall Resume Score</div>
            <div class="hero-sub">{R['hero_msg']}</div>
            <span class="hero-badge"
                  style="background:{gc}18; border-color:{gc}66; color:{gc};">
                📈 {R['grade']}
            </span>
            <div style="margin-top:12px; font-size:12px; color:#475569;">
                {"🤖 RF-enhanced score" if R['used_rf'] else "📐 Rule-based score"}
                &nbsp;·&nbsp; TF-IDF: {R['tfidf_sim']}
                &nbsp;·&nbsp; Semantic: {R['sem_sim']}
                &nbsp;·&nbsp; {R['wc']} words
            </div>
        </div>
        <div class="gauge-wrap">{gauge}</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION BREAKDOWN
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Section Breakdown")

    secs = R["sections"]

    # Row 1: Contact | Summary | Experience
    r1a, r1b, r1c = st.columns(3, gap="medium")
    r1a.markdown(render_card(secs["contact"]),    unsafe_allow_html=True)
    r1b.markdown(render_card(secs["summary"]),    unsafe_allow_html=True)
    r1c.markdown(render_card(secs["experience"]), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Row 2: Skills | Education (centred)
    _, r2a, r2b, _ = st.columns([0.4, 1, 1, 0.4], gap="medium")
    r2a.markdown(render_card(secs["skills"]),    unsafe_allow_html=True)
    r2b.markdown(render_card(secs["education"]), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ATS COMPATIBILITY
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎯 ATS Compatibility Score")

    a1, a2, a3, a4, a5 = st.columns(5, gap="small")
    for col, label, val, mx in [
        (a1, "🔑 Keyword Match",      R["kw_score"],  40),
        (a2, "🧠 Semantic Sim.",      R["sem_score"], 30),
        (a3, "🛠️ Skills Coverage",    R["sk_score"],  10),
        (a4, "📅 Experience",         R["exp_score"], 15),
        (a5, "📄 Formatting",         R["fmt_score"],  5),
    ]:
        pct = int(val / mx * 100)
        clr = score_color(pct)
        col.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-score" style="color:{clr};">{val}
                <span class="mini-max">/{mx}</span>
            </div>
            <div class="prog-bg" style="margin-top:8px;">
                <div class="prog-fill" style="width:{pct}%;background:{clr};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # KEYWORD & SKILLS GAP
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Keyword & Skills Gap")

    ka, sa = st.columns(2, gap="large")
    with ka:
        st.markdown(f"**Keywords** — `{len(R['matched_kw'])}` matched · `{len(R['missing_kw'])}` missing")
        with st.expander(f"✅ Keywords found in resume ({len(R['matched_kw'])})"):
            st.markdown(chips(R["matched_kw"][:30], "chip-g"), unsafe_allow_html=True)
        with st.expander(f"❌ Missing keywords — add these to your resume ({len(R['missing_kw'])})"):
            st.markdown(chips(R["missing_kw"], "chip-r"), unsafe_allow_html=True)

    with sa:
        st.markdown(f"**Skills** — `{len(R['res_skills'])}` found · `{len(R['miss_skills'])}` missing from JD")
        with st.expander(f"✅ Skills matched ({len([s for s in R['jd_skills'] if s in R['res_skills']])})"):
            st.markdown(chips([s for s in R["jd_skills"] if s in R["res_skills"]], "chip-g"), unsafe_allow_html=True)
        with st.expander(f"⚠️ Skills to add ({len(R['miss_skills'])})"):
            st.markdown(chips(R["miss_skills"], "chip-y"), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIENCE + FORMATTING
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📅 Experience & Formatting")

    ec, fc = st.columns(2, gap="large")
    with ec:
        req_s = f"{R['req_yrs']:.0f} years required" if R["req_yrs"] > 0 else "Not specified in JD"
        res_s = f"{R['res_yrs']:.0f} years detected"  if R["res_yrs"] > 0 else "Not detected"
        st.metric("JD Requirement",      req_s)
        st.metric("Detected in Resume",  res_s)
        if R["req_yrs"] > 0 and R["res_yrs"] < R["req_yrs"]:
            st.warning(f"Gap: ~{R['req_yrs']-R['res_yrs']:.0f} year(s) short. Include internships and projects.")
        elif R["req_yrs"] > 0:
            st.success("Experience requirement met ✓")

    with fc:
        st.markdown(f"**Formatting Score: {R['fmt_score']} / 5**")
        st.caption(f"Word count: {R['wc']}  ·  Special char ratio: ~{len(re.findall(r'[^a-zA-Z0-9 ]', val['text']))/max(len(val['text']),1):.1%}  ·  Tables: {'Yes' if val['tables'] else 'No'}")
        if R["fmt_issues"]:
            for issue in R["fmt_issues"]: st.warning(f"⚠ {issue}")
        else:
            st.success("✅ Formatting looks clean and ATS-friendly.")

    # ─────────────────────────────────────────────────────────────────────────
    # IMPROVEMENT CHECKLIST
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💡 Top Improvement Actions")

    tips_all = []
    for sec in R["sections"].values():
        if sec["score"] < 80:
            for tip in sec["tips"][:2]:
                tips_all.append((sec["title"], tip, sec["score"]))

    if R["fmt_issues"]:
        for issue in R["fmt_issues"]:
            tips_all.append(("Formatting", issue, int(R["fmt_score"]/5*100)))

    tips_all.sort(key=lambda x: x[2])  # lowest score first

    if not tips_all:
        st.success("🎉 Excellent! Your resume is well-optimised. Focus on tailoring your cover letter.")
    else:
        for section, tip, score in tips_all[:8]:
            clr = score_color(score)
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:12px;padding:10px 14px;'
                f'background:#0f172a;border-radius:8px;border-left:3px solid {clr};margin-bottom:8px;">'
                f'<span style="font-size:11px;font-weight:600;color:{clr};white-space:nowrap;'
                f'padding-top:2px;min-width:120px;">{section}</span>'
                f'<span style="font-size:14px;color:#94a3b8;line-height:1.55;">› {tip}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.caption(
        f"ATS Resume Scorer · "
        f"{'RF model active' if R['used_rf'] else 'Rule-based (place ats_models/ folder to enable RF)'} · "
        "No data stored externally."
    )

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;opacity:.4;">
        <div style="font-size:56px;">📄</div>
        <div style="font-size:20px;font-weight:600;margin-top:12px;">
            Upload a resume and paste a job description to begin
        </div>
        <div style="font-size:14px;margin-top:8px;">
            You'll get a full section-by-section breakdown just like a real ATS system.
        </div>
    </div>
    """, unsafe_allow_html=True)