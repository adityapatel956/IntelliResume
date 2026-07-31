import streamlit as st
from utils import extract_resume_text, fetch_all_jobs, match_resume_to_jobs, get_resume_suggestions
import pandas as pd
import html

st.set_page_config(page_title="IntelliResume", layout="wide")

st.markdown("""
<h2 style="
    margin-bottom:0;
    background: linear-gradient(90deg, #A78BFA, #F472B6, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
">
📄 IntelliResume
</h2>

<p style="
    color:#9CA3AF;
    font-size:18px;
    margin-top:5px;
">
Smart Resume Analysis & Job Matching
</p>
""", unsafe_allow_html=True)

st.markdown("""<style>
:root {
    --bg-main: #0F1117;
    --bg-card: #1A1D27;
    --bg-card-hover: #20232F;
    --accent: #8B5CF6;
    --accent-hover: #7C3AED;
    --accent-pink: #F472B6;
    --accent-yellow: #FBBF24;
    --accent-green: #34D399;
    --accent-blue: #60A5FA;
    --text-main: #EDEDF2;
    --text-muted: #9A9CAD;
    --border: #2A2D3A;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border);
}

/* Job card grid - responsive: auto-fills columns, stacks on mobile */
.job-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 18px;
    margin-top: 10px;
    margin-bottom: 30px;
}

.job-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    padding: 20px;
    border-radius: 14px;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.job-card:nth-child(4n+2) { border-top-color: var(--accent-pink); }
.job-card:nth-child(4n+3) { border-top-color: var(--accent-blue); }
.job-card:nth-child(4n+4) { border-top-color: var(--accent-green); }

.job-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    background-color: var(--bg-card-hover);
}

.job-card h3 {
    font-size: 17px;
    font-weight: 600;
    color: var(--text-main);
    margin: 0 0 10px 0;
    line-height: 1.3;
}

.score-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    background-color: rgba(139, 92, 246, 0.18);
    color: #C4B5FD;
    margin-bottom: 14px;
    width: fit-content;
}

.score-high { background-color: rgba(52, 211, 153, 0.18); color: #6EE7B7; }
.score-mid { background-color: rgba(251, 191, 36, 0.18); color: #FCD34D; }
.score-low { background-color: rgba(148, 163, 184, 0.18); color: #CBD5E1; }

.apply-button {
    display: inline-block;
    padding: 9px 18px;
    background: linear-gradient(90deg, var(--accent), var(--accent-pink));
    color: white !important;
    font-weight: 600;
    font-size: 14px;
    border-radius: 8px;
    text-decoration: none;
    text-align: center;
    transition: opacity 0.2s ease, transform 0.2s ease;
    width: fit-content;
}

.apply-button:hover {
    opacity: 0.85;
    transform: scale(1.03);
}

/* Resume text preview box */
.text-area {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    padding: 16px;
    border-radius: 10px;
}

/* Keyword chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 6px 0 18px 0;
}

.chip {
    background-color: rgba(139, 92, 246, 0.15);
    color: #C4B5FD;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 500;
}

.chip:nth-child(4n+2) { background-color: rgba(244, 114, 182, 0.15); color: #F9A8D4; }
.chip:nth-child(4n+3) { background-color: rgba(96, 165, 250, 0.15); color: #93C5FD; }
.chip:nth-child(4n+4) { background-color: rgba(251, 191, 36, 0.15); color: #FCD34D; }

/* Section spacing */
h2, h3 {
    margin-top: 28px;
}

/* Sidebar headers get a colorful accent */
section[data-testid="stSidebar"] h2 {
    color: var(--accent-pink);
    font-size: 18px;
}

/* Buttons (Get AI Suggestions, etc.) */
.stButton > button {
    background: linear-gradient(90deg, var(--accent), var(--accent-blue));
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity 0.2s ease;
}

.stButton > button:hover {
    opacity: 0.85;
    color: white;
}

/* Mobile responsiveness */
@media (max-width: 640px) {
    .job-grid {
        grid-template-columns: 1fr;
    }
    .job-card {
        padding: 16px;
    }
    h2 {
        font-size: 20px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("Job Filters")
remote = st.sidebar.checkbox("Only Remote Jobs")
company_filter = st.sidebar.text_input("Filter by Company Name (e.g., Microsoft, Google)")
tech_stack = st.sidebar.text_input("Filter by Tech Stack (e.g., Python, JavaScript)")

# Groq API key: prefer st.secrets (for deployed apps), fall back to
# a manual sidebar input (for local testing without a secrets file)

try:
    gemini_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    gemini_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com/keys"
    )


# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_resume_text(uploaded_file)

    if not resume_text.strip():
        st.error("⚠️ Couldn't read any text from this PDF. It may be a scanned image, "
                  "password-protected, or corrupted. Please try a different PDF (one with "
                  "selectable text works best).")
        st.stop()

    st.success("Resume text extracted!")

    st.subheader("Extracted Resume Content")
    st.markdown('<div class="text-area">', unsafe_allow_html=True)
    st.text_area("Resume Text", resume_text, height=250, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Fetching live job listings from multiple sources..."):
        jobs_df = fetch_all_jobs()

    # Make sure required columns exist so the app doesn't crash on
    # sources that are missing fields
    for col in ["title", "company", "description", "url"]:
        if col not in jobs_df.columns:
            jobs_df[col] = ""
    jobs_df["company"] = jobs_df["company"].fillna("")
    jobs_df["title"] = jobs_df["title"].fillna("Untitled Role")

    # Apply filters BEFORE matching
    if not jobs_df.empty:
        if remote:
            if 'remote' in jobs_df.columns:
                jobs_df = jobs_df[jobs_df['remote'] == True]
        if company_filter:
            jobs_df = jobs_df[jobs_df['company'].str.contains(company_filter, case=False, na=False)]
        if tech_stack:
            if 'tech_stack' in jobs_df.columns:
                jobs_df = jobs_df[jobs_df['tech_stack'].str.contains(tech_stack, case=False, na=False)]

        with st.spinner("Matching your resume with available jobs..."):
            matched_jobs = match_resume_to_jobs(resume_text, jobs_df)

        if not matched_jobs.empty:
            st.subheader("💼 Top Job Matches for You")
            card_parts = ['<div class="job-grid">']
            for _, row in matched_jobs.iterrows():
                safe_title = html.escape(str(row['title']))
                safe_company = html.escape(str(row['company']))
                safe_url = html.escape(str(row['url']), quote=True)
                match_pct = row["similarity"] * 100
                score_class = "score-high" if match_pct >= 60 else "score-mid" if match_pct >= 40 else "score-low"
                card_parts.append(
                    f'<div class="job-card">'
                    f'<div>'
                    f'<h3>{safe_title}</h3>'
                    f'<p style="color: var(--text-muted); margin: 0 0 12px 0; font-size: 14px;">{safe_company}</p>'
                    f'<div class="score-badge {score_class}">Match: {match_pct:.0f}%</div>'
                    f'</div>'
                    f'<a class="apply-button" href="{safe_url}" target="_blank">🔗 Apply Now</a>'
                    f'</div>'
                )
            card_parts.append('</div>')
            cards_html = "".join(card_parts)
            st.markdown(cards_html, unsafe_allow_html=True)

            # ---- AI Resume Suggestions (Groq) ----
            st.subheader("🤖 AI Resume Suggestions")
            if not gemini_api_key:
                st.info("Add your free Groq API key in the sidebar to get personalized "
                        "resume improvement suggestions (general + tailored to your "
                        "top job match). Get one at https://console.groq.com/keys — "
                        "no card or billing needed.")
            else:
                if st.button("Get AI Suggestions"):
                    with st.spinner("Analyzing your resume with AI..."):
                        top_job = matched_jobs.iloc[0].to_dict()
                        result = get_resume_suggestions(
                            resume_text, top_job=top_job, api_key=gemini_api_key
                        )

                    if "error" in result:
                        st.warning(f"⚠️ Error: {result['error']}")
                    else:
                        col1, col2 = st.columns(2)
                        col1.metric("ATS Score", f"{result.get('atsScore', 0)}/100")
                        col2.metric("Job Match Score", f"{result.get('matchScore', 0)}/100")

                        if result.get("strengths"):
                            st.markdown("**✅ Strengths**")
                            for s in result["strengths"]:
                                st.markdown(f"- {s}")

                        if result.get("weaknesses"):
                            st.markdown("**⚠️ Weaknesses**")
                            for w in result["weaknesses"]:
                                st.markdown(f"- {w}")

                        if result.get("missingKeywords"):
                            st.markdown("**🔑 Missing Keywords**")
                            chips_html = '<div class="chip-container">'
                            for kw in result["missingKeywords"]:
                                chips_html += f'<span class="chip">{html.escape(str(kw))}</span>'
                            chips_html += '</div>'
                            st.markdown(chips_html, unsafe_allow_html=True)

                        if result.get("suggestions"):
                            st.markdown("**💡 Suggestions**")
                            for sug in result["suggestions"]:
                                st.markdown(f"- {sug}")
        else:
            st.warning("⚠️ No jobs matched your resume after applying filters.")
    else:
        st.warning("⚠️ Could not fetch jobs. Try again later.")

else:
    st.info("Upload a PDF resume to get started.")