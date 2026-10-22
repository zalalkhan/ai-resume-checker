# 📄 ATS Resume Checker

Upload a resume (PDF or DOCX) and get an **ATS score out of 100**, a section-by-section
breakdown, missing keywords, prioritised improvements and rewritten bullet examples.
Optionally paste a job description for a tailored score.

Built with **Streamlit** and **Google Gemini Flash**.

## Features
- PDF and DOCX parsing (including tables in DOCX)
- ATS score + 6 sub-scores (formatting, keywords, experience, skills, education, readability)
- Strengths, weaknesses, missing keywords, prioritised fixes
- Before/after bullet rewrites
- Instant rule-based checks (email, phone, LinkedIn, sections, word count)
- Downloadable JSON report

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then either:

**Option A - secrets file** (recommended). Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-3.8-flash"   # optional
```

**Option B** - leave it empty and paste the key into the app's sidebar.

Start the app:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (do **not** commit `.streamlit/secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, pick your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste: `GEMINI_API_KEY = "your-key-here"`
5. Click **Deploy**.

## Project structure
```
.
├── app.py
├── requirements.txt
└── README.md
```

## Notes
- Scanned/image-only PDFs have no extractable text and will be rejected. Use a text-based PDF or DOCX.
- The ATS score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Resumes are sent to the Gemini API. Do not upload documents you are not comfortable sharing.
- Model names change over time. The app defaults to `gemini-3.8-flash` and automatically falls back to other Flash models if it gets a 404; you can also set `GEMINI_MODEL` yourself.
