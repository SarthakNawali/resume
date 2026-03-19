"""
IT ENGINEER ATS SYSTEM - STREAMLIT APP
======================================
Features:
- Resume upload + rejection count input
- Groq API for intelligent recommendations
- No job description needed

Install: pip install streamlit pdfplumber scikit-learn numpy pandas joblib groq
Run: streamlit run app.py
"""

import os, re, io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import pdfplumber
import streamlit as st
from groq import Groq

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IT Engineer ATS",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════════════════════
# STYLING
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.header-main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.header-title { font-size: 32px; font-weight: 800; margin: 0; }
.header-sub { font-size: 14px; opacity: 0.9; margin: 5px 0 0; }

.metric-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
.metric-value { font-size: 28px; font-weight: 800; color: #333; }

.section-header {
    font-size: 18px; font-weight: 700; color: #333;
    margin-top: 20px; margin-bottom: 15px;
    padding-bottom: 10px; border-bottom: 2px solid #667eea;
}

.skill-chip {
    display: inline-block;
    padding: 6px 12px; margin: 4px 4px 4px 0;
    border-radius: 20px; font-size: 12px; font-weight: 500;
}
.skill-found { background: #d1fae5; color: #065f46; }
.skill-missing { background: #fee2e2; color: #991b1b; }

.alert {
    padding: 15px; border-radius: 8px; margin: 10px 0;
}
.alert-info { background: #eff6ff; border-left: 4px solid #0284c7; color: #0c2340; }
.alert-warning { background: #fffbeb; border-left: 4px solid #f59e0b; color: #92400e; }
.alert-success { background: #ecfdf5; border-left: 4px solid #10b981; color: #065f46; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean text."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\x80-\xFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_pdf_text(pdf_file) -> str:
    """Extract text from PDF."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text
    except Exception as e:
        st.error(f"❌ Failed to parse PDF: {e}")
        return ""

def load_trained_model():
    """Load model."""
    model_path = "ats_models/rf_it_engineer_classifier.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.info("Run the training notebook first.")
        st.stop()
    return joblib.load(model_path)

def extract_it_skills(text: str, all_it_skills: dict) -> list:
    """Extract IT skills."""
    found = []
    for skill in all_it_skills.keys():
        if re.search(r'\b' + skill + r'\b', text, re.IGNORECASE):
            found.append(skill)
    return list(set(found))

def extract_years_experience(text: str) -> float:
    """Extract years."""
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\s*years?\s+(?:in|of)\s+',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return float(max(matches))
    return 0.0

def predict_rejection_risk(resume_text: str, model_data: dict) -> dict:
    """Predict rejection risk."""
    rf_model = model_data['model']
    it_skills = model_data['it_skills']
    all_it_skills = model_data['all_it_skills']
    feature_cols = model_data['feature_cols']
    
    clean = clean_text(resume_text)
    found_skills = extract_it_skills(clean, all_it_skills)
    
    skill_counts = {cat: 0 for cat in it_skills.keys()}
    for skill in found_skills:
        if skill in all_it_skills:
            cat = all_it_skills[skill]
            skill_counts[cat] += 1
    
    years_exp = extract_years_experience(clean)
    word_count = len(clean.split())
    
    feature_vector = np.array([[
        len(found_skills),
        years_exp,
        word_count,
        *[skill_counts[cat] for cat in it_skills.keys()]
    ]])
    
    pred = int(rf_model.predict(feature_vector)[0])
    probs = rf_model.predict_proba(feature_vector)[0]
    confidence = float(probs.max())
    
    missing = {cat: [] for cat in it_skills.keys()}
    for cat, skills in it_skills.items():
        for skill in skills:
            if skill not in found_skills:
                missing[cat].append(skill)
    
    return {
        'rejection_risk': pred,
        'confidence': confidence,
        'skills_found': found_skills,
        'missing_skills': missing,
        'skill_counts': skill_counts,
        'years_experience': years_exp,
        'word_count': word_count,
        'total_skills_found': len(found_skills),
    }

def get_groq_recommendations(result: dict, user_rejection_count: int, api_key: str) -> str:
    """Get Groq AI recommendations."""
    try:
        client = Groq(api_key=api_key)
        
        # Build context
        skills_str = ', '.join(result['skills_found']) if result['skills_found'] else 'None'
        missing_str = ""
        for cat in ['Languages', 'Cloud & DevOps', 'Databases', 'Frameworks']:
            if result['missing_skills'].get(cat):
                missing_str += f"\n{cat}: {', '.join(result['missing_skills'][cat][:3])}"
        
        prompt = f"""You are an expert IT recruiter providing resume feedback.

CANDIDATE PROFILE:
- Skills Found: {skills_str}
- Years Experience: {result['years_experience']:.0f}
- Resume Length: {result['word_count']} words
- Total IT Skills: {result['total_skills_found']}

REJECTION HISTORY:
- Candidate reported: {user_rejection_count} rejection(s)
- ML Model predicts: {result['rejection_risk']} rejections

MISSING SKILLS:{missing_str}

Provide SPECIFIC, ACTIONABLE feedback:
1. Top 3 MUST-ADD skills (with concrete reasoning)
2. 5 concrete improvement steps for the resume
3. Estimated timeline for improvements
4. How to showcase these improvements

Be encouraging but direct. Focus on highest-impact changes."""
        
        # Using current available model: llama-3.1-70b-versatile
        message = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nTroubleshooting:\n1. Check API key is valid\n2. Visit https://console.groq.com to verify\n3. Make sure you have API credits available"

# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════
try:
    model_data = load_trained_model()
    it_skills = model_data['it_skills']
    all_it_skills = model_data['all_it_skills']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-main">
    <div class="header-title">💼 IT Engineer ATS System</div>
    <div class="header-sub">Resume Analysis with Groq AI Recommendations</div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR - INPUTS
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📤 Upload Resume")
    pdf_file = st.file_uploader("Select PDF", type="pdf", label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📊 Rejection Count")
    st.write("How many times rejected?")
    rejection_count = st.radio(
        "Choose:",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1️⃣ Once (minor gaps)",
            2: "2️⃣ Twice (moderate gaps)",
            3: "3️⃣ Three times (major gaps)",
            4: "4️⃣ Four+ times (critical gaps)"
        }.get(x),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🔑 Groq API Key")
    groq_api_key = st.text_input(
        "Enter key:",
        type="password",
        placeholder="gsk_...",
        help="Free at https://console.groq.com",
        label_visibility="collapsed"
    )
    
    if not groq_api_key:
        st.info("Get free API key at https://console.groq.com")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

if pdf_file:
    with st.spinner("📖 Analyzing resume..."):
        resume_text = extract_pdf_text(pdf_file)
        
        if resume_text:
            result = predict_rejection_risk(resume_text, model_data)
            
            # ANALYSIS SUMMARY
            st.markdown('<div class="section-header">📊 Analysis Summary</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Your Count", rejection_count)
            with col2:
                st.metric("ML Predicts", result['rejection_risk'])
            with col3:
                st.metric("Skills Found", result['total_skills_found'])
            with col4:
                st.metric("Experience", f"{result['years_experience']:.0f}y")
            
            # SKILLS BREAKDOWN
            st.markdown('<div class="section-header">🛠️ Skills Breakdown</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Found Skills:**")
                if result['skills_found']:
                    st.markdown(" ".join(
                        f'<span class="skill-chip skill-found">{s}</span>'
                        for s in sorted(result['skills_found'])
                    ), unsafe_allow_html=True)
                else:
                    st.warning("No IT skills detected")
            
            with col2:
                st.write("**Skills by Category:**")
                for cat, count in result['skill_counts'].items():
                    if count > 0:
                        st.write(f"• {cat}: {count}")
            
            # AI RECOMMENDATIONS
            st.markdown('<div class="section-header">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)
            
            if groq_api_key:
                with st.spinner("💭 Generating recommendations from Groq AI..."):
                    recommendations = get_groq_recommendations(result, rejection_count, groq_api_key)
                    st.markdown(recommendations)
            else:
                st.markdown("""
                <div class="alert alert-info">
                    <strong>💡 Get AI recommendations:</strong><br>
                    Enter your Groq API key above to receive personalized recommendations.
                    Get free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a>
                </div>
                """, unsafe_allow_html=True)
            
            # MISSING SKILLS
            st.markdown('<div class="section-header">📚 Skills to Add</div>', unsafe_allow_html=True)
            
            for cat in ['Languages', 'Databases', 'Cloud & DevOps', 'Frameworks', 'Data & ML', 'Architecture']:
                if cat in result['missing_skills']:
                    missing_list = result['missing_skills'][cat]
                    if missing_list:
                        st.write(f"**{cat}:**")
                        st.markdown(" ".join(
                            f'<span class="skill-chip skill-missing">{s}</span>'
                            for s in missing_list[:5]
                        ), unsafe_allow_html=True)
        else:
            st.error("❌ Could not extract text from PDF")
else:
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px; opacity: 0.5;">
        <div style="font-size: 64px; margin-bottom: 20px;">📄</div>
        <div style="font-size: 24px; font-weight: 600; margin-bottom: 10px;">
            Upload Resume to Start
        </div>
        <div style="font-size: 14px;">
            👈 Use the sidebar to upload and analyze
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("💼 IT Engineer ATS • Resume Analysis • Groq AI Powered")