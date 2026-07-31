# 📄 AI-Powered Resume Analyzer & Multi-Source Job Matcher

An AI-powered web app that:

- Analyzes your uploaded resume (PDF)
- Fetches live jobs from multiple sources
- Matches the best job opportunities based on your skills and experience
- Uses an LLM to give you personalized resume improvement suggestions, scored and structured

Built using **Streamlit**, **Python**, **Pandas**, **Sentence-Transformers**, and **Groq (LLM API)** for smart, semantic matching and AI-generated feedback.

---

## 🔥 Features

- 📄 Upload your Resume (PDF)
- 🧠 Automatic Resume Text Extraction (PDFMiner), with graceful handling of scanned/corrupted PDFs
- 🌎 Fetch Jobs from Multiple Sources
  - RemoteOK (API)
  - Arbeitnow (API)
- 🤖 Smart Resume-to-Job Matching
  - Sentence-transformer embeddings (`all-MiniLM-L6-v2`) on full resume/job text (no stemming - preserves context for better accuracy)
  - Cosine similarity to rank best matches
- 🧠 AI Resume Suggestions (Groq - free, no billing required)
  - Returns structured JSON: ATS score, job match score, strengths, weaknesses, missing keywords, and specific suggestions
  - Scraped job descriptions are sanitized before being sent to the LLM, to strip out any hidden/malicious instructions embedded in listings (prompt-injection protection)
- 🎯 Job Filters (in sidebar)
  - Only Remote Jobs
  - Filter by Company Name
  - Filter by Tech Stack (e.g., Python, React)
- 🌗 Dark Mode UI with modern job cards (title, company, similarity score, Apply button)
- ⚡ Cached model loading and job fetching, so repeat uploads are fast

---

## 📥 Installation

1. **Clone the repository**

```
git clone https://github.com/TR-3N/-AI-Powered-Resume-Analyzer-Multi-Source-Job-Matcher.git
cd -AI-Powered-Resume-Analyzer-Multi-Source-Job-Matcher
```

2. **Create & activate virtual environment (recommended)**

```
python -m venv venv
# Windows (PowerShell)
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

This installs Streamlit, Pandas, scikit-learn, pdfminer.six, BeautifulSoup, sentence-transformers, sqlalchemy, groq, and NLTK.

4. **Get a free Groq API key (for AI resume suggestions)**

- Go to https://console.groq.com/keys
- Sign in with Google/GitHub
- Click "Create API Key" and copy it
- No card or billing required

You'll paste this into the app's sidebar when it's running (or set it as `GROQ_API_KEY` in a `.streamlit/secrets.toml` file if deploying).

---

## 🚀 Running the App

From the project folder (with venv active):

```
streamlit run app.py
```

- This starts a local server (usually `http://localhost:8501`).
- In the browser:
  - Upload your resume (PDF).
  - The app extracts text, fetches jobs, matches them, and shows top job cards with similarity scores and "Apply Now" links.
  - Paste your Groq API key in the sidebar, then click "Get AI Suggestions" to see your ATS score, match score, strengths, weaknesses, missing keywords, and suggestions.

---

## 🛠 Project Structure

```
.
├── app.py            # Main Streamlit app (UI, filters, rendering)
├── utils.py          # Resume extraction, job fetching, matching logic, AI suggestions
├── job_scraper.py    # Optional: scrape Indeed + insert into SQLite
├── init_db.py         # Optional: initialize jobs.db schema
├── jobs.csv           # Optional: sample/static jobs data
├── requirements.txt
└── README.md
```

---

## ✨ Future Improvements

- Better salary estimation
- Apply to jobs with a single click
- One-click cover letter generation using the same AI suggestions
- Additional job sources

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open an Issue or Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.