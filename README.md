<div align="center">

# 🧠 CodeSense
### AI-Based Code Quality Analyzer

*My B.Tech Final Year Project — built from scratch over 6 months 😅*

<br>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)

<br>

![GitHub last commit](https://img.shields.io/github/last-commit/ShivamJaiswal1111/CodeSense-AI-Based-Code-Quality-Analyzer?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/ShivamJaiswal1111/CodeSense-AI-Based-Code-Quality-Analyzer?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/ShivamJaiswal1111/CodeSense-AI-Based-Code-Quality-Analyzer?style=flat-square)

<br>

> **"I was tired of submitting code and not knowing WHY it was bad. So I built something that actually explains it."**

<br>

[🚀 Quick Start](#-quick-start) • [✨ Features](#-what-it-does) • [📸 Screenshots](#-screenshots) • [🏗️ How it Works](#%EF%B8%8F-how-it-works) • [🤝 Contributing](#-contributing)

</div>

---

## 👋 Hey, what's this?

So basically in my 3rd year I kept getting feedback like *"your code quality is bad"* from professors but nobody ever told me **what exactly** was wrong or **how to fix it**. That's when I decided to make CodeSense for my final year project.

It analyzes your Python, Java, or C++ code and gives you:
- A proper score (not just vibes)
- Exactly what's wrong and on which line
- How to fix it
- Which algorithms you used and their complexity
- Security vulnerabilities you didn't even know you had

I spent way too many nights on this. Hope it helps someone 🙏

---

## ✨ What it does

<table>
<tr>
<td width="50%">

### 🤖 ML-Based Scoring
Trained an ensemble model (Random Forest + Gradient Boosting) on 10,000 code samples. The score actually comes from the model — not some random formula I made up. R² ≥ 0.90 on test set.

</td>
<td width="50%">

### 🧠 Algorithm Detection
Detects 40+ algorithms (bubble sort to Dijkstra's) and tells you the Big-O complexity. Honestly this part was the most fun to build.

</td>
</tr>
<tr>
<td width="50%">

### 🔒 Security Scanner
50+ vulnerability patterns. Finds SQL injection, hardcoded passwords, weak hashing and more. Saved me from some embarrassing code lol.

</td>
<td width="50%">

### 📚 Learning Path
Doesn't just point out problems — gives you resources, LeetCode problems, and step-by-step improvement tips based on YOUR specific issues.

</td>
</tr>
<tr>
<td width="50%">

### 🔧 Auto-Fix Suggestions
Shows you the before/after for safe fixes. Doesn't just say "this is bad" — actually shows what good looks like.

</td>
<td width="50%">

### 📈 Progress Tracking
Tracks your scores over time so you can actually see yourself improving. Made this because I wanted to see if I was getting better.

</td>
</tr>
</table>

---

## 📸 Screenshots

### Login & Register Page
<img 
  src="https://drive.google.com/uc?export=view&id=1ZwSK5XwSvUIxGwtO5OIs_7xkSJJc0D9-" 
  alt="Login & Register Page" 
  width="500"
/>

### Dashboard
<img 
  src="https://drive.google.com/uc?export=view&id=1XQBL_chZZ7a1yH1XqoibwOXijPWONL9C" 
  alt="Dashboard" 
  width="500"
/>

### Analysis Results
<img 
  src="https://drive.google.com/uc?export=view&id=1qN-EBEIuPzhOyLCHmAdNh-zRHpRNrh8D" 
  alt="Analysis Results" 
  width="500"
/>

### Achievement
<img 
  src="https://drive.google.com/uc?export=view&id=1A_VLJHkzVrpsbkpmEnz0z5oUtuhd5LRB" 
  alt="Achievement" 
  width="500"
/>

### Progress
<img 
  src="https://drive.google.com/uc?export=view&id=1bLUcJ6jiZ-eKHkDXIeaqYDs2r3JyUrtZ" 
  alt="Progress" 
  width="500"
/>

---

## 🚀 Quick Start

> Tested on Windows 11 and Ubuntu 22.04. Should work on Mac too but I don't have one to test 😅

### Step 1 — Clone the repo

```bash
git clone https://github.com/ShivamJaiswal1111/CodeSense-AI-Based-Code-Quality-Analyzer
cd CodeSense-AI-Based-Code-Quality-Analyzer
```

### Step 2 — Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This installs scikit-learn, streamlit, plotly, bcrypt etc. Might take 2-3 mins on slow internet.

### Step 4 — Setup config

```bash
cp .env.example .env
```

Open `.env` and change the `SECRET_KEY` to anything random (min 32 chars). 

### Step 5 — Train the ML model

```bash
python train_model.py --samples 10000
```

This runs once and saves the model to `models/`. Takes about 30-60 seconds. You'll see output like:

```
Generating 10000 training samples...
Running 10-fold cross-validation...
CV R²: 0.923 ± 0.008  ✅ PASSED
Model saved to models/codesense_model.pkl
```

### Step 6 — Run it! 🎉

```bash
streamlit run app.py
```

Go to **http://localhost:8501** — create an account and start analyzing!

---

## 🏗️ How it Works

I know most project READMEs skip this part but I think it's actually interesting so here's a quick breakdown:

```
Your Code
    │
    ├──► Syntax Analyzer    (AST + tokenizer, finds parse errors)
    ├──► Semantic Analyzer  (unused vars, dead code, mutable defaults)
    ├──► Static Analyzer    (security patterns, complexity, style)
    ├──► DSA Detector       (pattern matching for 40+ algorithms)
    └──► Context Engine     (is this a test file? web app? algorithm?)
                │
                ▼
         Feature Extractor
         (converts everything into 33 numbers the ML model understands)
                │
                ▼
         ML Model (RandomForest + GradientBoosting ensemble)
                │
                ▼
         Quality Score (0-100) + Grade + Confidence
                │
                ▼
         Feedback Engine (generates human-readable explanation)
```

The score comes ENTIRELY from the ML model. I spent a long time making sure there are no hardcoded rules that override it.

### The 33 Features

The ML model uses 33 features I engineered from the static analysis:

| Category | Features | Examples |
|---|---|---|
| Basic Metrics | 8 | Lines of code, comment ratio, blank lines |
| Complexity | 6 | Cyclomatic complexity, nesting depth, function length |
| Style | 5 | Naming consistency, magic numbers, long lines |
| Security | 4 | Issue count, critical count, severity score |
| DSA | 3 | Algorithm complexity score, count, DS count |
| Contextual | 7 | Code duplication, exception coverage, technical debt |

---

## 📁 Project Structure

```
CodeSense/
│
├── 📱 app.py                 ← Main Streamlit app (all the pages)
├── 🔍 analyzer.py            ← Static analysis engine
├── 🧠 dsa_detector.py        ← Algorithm + data structure detection
├── 📊 features.py            ← Feature engineering (33 features)
├── 🤖 train_model.py         ← ML model training + inference
├── 💬 student_feedback.py    ← Feedback generation
├── ⚠️  syntax_analyzer.py    ← Syntax error detection
├── 🔬 semantic_analyzer.py   ← Semantic issue detection
├── 🌍 context_engine.py      ← Code context understanding
├── 🔧 code_fixer.py          ← Auto-fix suggestions
├── 🎨 ui_components.py       ← Reusable UI components
│
├── 🗄️  db.py                 ← SQLite database (WAL mode)
├── 🔐 auth.py                ← Auth (bcrypt + OTP + sessions)
├── ⚡ cache.py               ← Two-tier LRU + disk cache
├── ✅ validators.py          ← Input validation
├── 🛠️  utils.py              ← GitHub fetch, exports, timing
├── 📝 logger.py              ← Structured logging
├── ⚙️  config.py             ← Config management
├── 📌 constants.py           ← All constants in one place
│
├── 📁 models/                ← Saved ML model files
├── 📁 tests/                 ← Test suite (50+ tests)
├── 📁 logs/                  ← App logs
├── 📁 cache/                 ← Disk cache files
│
├── 📋 requirements.txt
└── 🔒 .env.example
```

---

## 🤖 ML Model Details

> This was my main challenge — making a model that actually gives meaningful scores

- **Algorithm:** Voting ensemble (RandomForest 55% + GradientBoosting 45%)
- **Training data:** 10,000 synthetic code samples with realistic score distributions
- **Validation:** 10-fold cross-validation (not just train/test split)
- **Target R²:** ≥ 0.90 (retrains if it doesn't hit this)
- **Contextual adjustment:** Max ±5 points from context engine (test files, algorithm-heavy code etc.)

I initially tried just using cyclomatic complexity as the score (lol) but that was terrible. The ML approach captures way more nuance.

---

## 🔒 Security Features

Both in the app AND in what it detects:

**App security:**
- bcrypt password hashing (12 rounds)
- OTP email verification
- Brute force lockout (5 attempts → 30 min lock)
- Parameterized SQL queries (no injection)
- Session tokens (48-byte cryptographically secure)

**Code security detection:**
- SQL injection, command injection, XSS
- Hardcoded passwords, API keys, tokens
- Weak crypto (MD5, SHA1, insecure random)
- Unsafe deserialization (pickle, yaml.load)
- Buffer overflows in C++ (gets, strcpy, sprintf)
- And 40+ more patterns

---

## 🧪 Running Tests

```bash
# Run all tests
python tests/test_suite_enhanced.py

# Or with pytest for better output
pip install pytest pytest-cov
pytest tests/ -v --tb=short --cov=. --cov-report=term-missing
```

Tests cover all major modules — syntax analysis, ML model, database, auth, cache, DSA detection etc.

---

Then go to **http://localhost:8501**

---

## 📚 Tech Stack

| What | Why I chose it |
|---|---|
| **Streamlit** | Fastest way to build a Python web UI. Perfect for data science projects. |
| **scikit-learn** | Industry standard ML library. The ensemble models are solid. |
| **SQLite** | No external database server needed. WAL mode handles concurrent reads fine. |
| **bcrypt** | Proper password hashing. None of that MD5 nonsense. |
| **Plotly** | Interactive charts that actually look good in dark mode. |
| **Python AST** | Built-in library for parsing Python code — no external parsers needed. |

---

## 🤝 Contributing

This is my final year project so I'm not taking major contributions right now, but if you find bugs or have suggestions feel free to open an issue!

If you fork this for your own project, a star ⭐ would be really appreciated — it helps with the placement portfolio 😄

---

## 📝 Known Issues / TODO

- [ ] Add support for JavaScript / TypeScript
- [ ] Fix occasional false positives in C++ template code detection  
- [ ] Add side-by-side code diff view in the Fixes tab
- [ ] Mobile responsive layout (Streamlit has some limitations here)
- [ ] Export analysis as PDF
- [ ] Add more LeetCode problem suggestions

---

## 🙏 Acknowledgements

- My project guide for not giving up on me when I showed up with "I'll just use cyclomatic complexity as the score" in month 2
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com) — referenced constantly
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) — for the security patterns
- [Streamlit docs](https://docs.streamlit.io) — surprisingly good docs
- Stack Overflow — obviously

---

<div align="center">

**Made with way too much coffee ☕ and late nights 🌙**

*Shivam Jaiswal — B.Tech CSE, Final Year*
*Sarthak Agrawal — B.Tech CSE, Final Year*
*Yash Sharma — B.Tech CSE, Final Year*
*Tanishq Kumar— B.Tech CSE, Final Year*

⭐ Star this repo if it helped you! ⭐

</div>
