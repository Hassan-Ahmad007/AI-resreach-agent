# 🔎 AI Research Agent

A beginner-friendly single-agent research app built with:

- **[CrewAI](https://docs.crewai.com)** — orchestrates the AI agent
- **[Groq](https://console.groq.com)** — runs the LLM (`openai/gpt-oss-120b`), very fast and has a free tier
- **[DuckDuckGo Search](https://pypi.org/project/ddgs/)** (`ddgs` library) — free web search, no API key needed
- **[Streamlit](https://streamlit.io)** — the web UI

You type a topic, the agent searches the web for it, and it writes you a
Markdown report with sources. The Groq API key is read **only** from
Streamlit Cloud's Secrets — there's no key-entry box in the app itself.

## 📁 Project structure

```
ai-research-agent/
├── app.py                        # Streamlit UI (reads GROQ_API_KEY from secrets)
├── agent.py                      # CrewAI agent, task, search tool
├── requirements.txt              # Python dependencies
├── .gitignore                    # keeps secrets.toml out of GitHub
├── .streamlit/
│   └── secrets.toml.example      # reference for what to paste into Streamlit Cloud
└── README.md                     # this file
```

---

## 1. Get a free Groq API key

1. Go to <https://console.groq.com/keys>
2. Sign up / log in
3. Click **Create API Key** and copy it somewhere safe (you'll only see it once)

---

## 2. Upload this project to GitHub

1. Create a new **empty** repository on GitHub (don't let it auto-add a README — you already have one here).
2. On the repo page, click **Add file → Upload files**, then drag in every file and folder from this project (including the hidden `.gitignore` and the `.streamlit` folder).
3. Commit the upload.

*(If you prefer the command line instead of the web upload, that works too — `git init`, `git add .`, `git commit -m "Initial commit"`, `git remote add origin <your-repo-url>`, `git push -u origin main`.)*

Your real API key never goes into this repo — it only lives in Streamlit Cloud's Secrets, set up in the next step.

---

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and log in with your GitHub account.
2. Click **New app**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Before clicking Deploy, open **Advanced settings**:
   - Set the **Python version** to `3.11`.
   - Under **Secrets**, paste:
     ```toml
     GROQ_API_KEY = "your_real_groq_api_key_here"
     ```
     (This is exactly the format shown in `.streamlit/secrets.toml.example`.)
5. Click **Deploy**.

Streamlit Cloud installs everything from `requirements.txt` and starts
`app.py`. After a minute or two your app goes live at a URL like
`https://your-app-name.streamlit.app`, and it will already have access to
your Groq key — no one visiting the app needs to enter anything.

**To update the key later**, or add new ones: open your app on Streamlit
Cloud → **⋮ menu → Settings → Secrets**, edit, and save (the app restarts
automatically).

---

## How it works (quick tour)

- **`agent.py`**
  - `duckduckgo_search` — a custom CrewAI **tool**: a Python function wrapped
    in `@tool(...)` so the agent can call it. Uses the free `ddgs` library.
  - `build_crew()` — creates one `Agent` (role: "Senior Research Analyst"),
    gives it the search tool and a Groq-powered brain
    (`LLM(model="groq/openai/gpt-oss-120b")`), and one `Task` describing
    exactly what the report should contain.
  - `run_research()` — the function `app.py` calls; builds the crew and
    returns the final report as a string.

- **`app.py`**
  - Reads `GROQ_API_KEY` from `st.secrets` only.
  - A simple form: topic input → button → spinner while the agent works →
    rendered Markdown report → download button.

## Customizing it

- **Change the model**: edit `model="groq/openai/gpt-oss-120b"` in
  `agent.py`. Any model on [Groq's model list](https://console.groq.com/docs/models)
  works — just prefix it with `groq/`.
- **Change report length/style**: edit the `description` and
  `expected_output` text inside `research_task` in `agent.py`.
- **Add a second agent** (e.g. a "Writer" who polishes the researcher's
  draft): add another `Agent`, another `Task` that depends on the first
  one's output (via `context=[research_task]`), and include both in the
  `Crew(agents=[...], tasks=[...])` lists.

## Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar shows "GROQ_API_KEY secret not found" | Go to your app on Streamlit Cloud → **⋮ → Settings → Secrets** and add `GROQ_API_KEY = "..."`, then save (app auto-restarts). |
| Deployment fails / app won't start | Open **Manage app → logs** on Streamlit Cloud. Most issues are a missing/misformatted secret or a Python version mismatch — set Python to 3.11 in Advanced settings. |
| `ModuleNotFoundError` during deploy | Double-check `requirements.txt` was uploaded to the repo root and isn't misspelled. |
| Search tool returns no results / errors | DuckDuckGo occasionally rate-limits automated queries. Wait a moment and try again, or reduce how many searches the agent does per run. |
