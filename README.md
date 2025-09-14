---
title: Tell Me
emoji: "💬🌿"
colorFrom: indigo
colorTo: green
sdk: streamlit
app_file: fresh_app_v2.py
pinned: false
tags:
- streamlit
short_description: Mental wellbeing chat (research)
---

**🌿 Tell Me — A Mental Well-Being Space**

Tell Me is a safe space for individuals seeking some well-being advice or a self-reflection space. It also provides the research community to simulate some LLM generated client-therapist synthetic data. This is a research prototype, not a medical device.

## Key Components of Tell Me:***

- **Tell Me Assistant**
  
  Tell Me Assistant is a Mental Well-being Chatbot designed to help individuals process their thoughts and emotions in a supportive way.
  It is not a substitute for professional care, but it offers a safe space for conversation and self-reflection.
  The Assistant is created with care, recognizing that people may turn to it during moments of initial support. Its goal is to make such therapeutic-style interactions more accessible and approachable for everyone.

  `fresh_app_v2.py` interconnected with `rag.py` and `llm_models.py` to power the Assistant with context using RAG
 
- **Simulate a Conversation**  
  This generates a synthetic client–therapist conversation from a short client profile. It helps create sample data for research and lets professionals inspect the dialogue quality. Outputs are created by an LLM and can guide future fine-tuning or evaluation.
  Multi‑turn, role‑locked *Therapist ↔ Client* dialogue from a brief persona (see `llm_models.py`).

- **Well‑being Planner (CrewAI)**  
  1) Transcript analysis (themes, emotions, triggers)  
  2) **7‑day plan** (CBT/behavioral steps, routines, sleep hygiene, social micro‑actions)  
  3) **Guided meditation** script + **MP3** (gTTS/Edge/Coqui/ElevenLabs)  
  Implemented in `crew_ai.py`, surfaced in the **Planner** tab in `fresh_app_v2.py`.


- **Evaluation**  
  Use `prep_responses.py` and `judge.py` to prepare and score generations using LLM as a judge and also the results of conducted Human Evaluation; see `Results/` for artifacts (e.g., *gpt4o/5 eval*).

---

## Repository Structure

```
.
├─ Results/                 # Evaluation outputs / artifacts (e.g., gpt4o eval)
├─ index_storage/           # Vector index built by rag.py
├─ rag_data/                # Source docs for RAG
├─ src/                     # Streamlit template seed
├─ bg.jpg                   # App background
├─ config.toml              # Streamlit config (dark mode default, etc.)
├─ crew_ai.py               # CrewAI pipeline (planner + meditation TTS)
├─ fresh_app_v2.py          # Main Streamlit app
├─ judge.py                 # Evaluation judge
├─ llm_models.py            # Prompt builders + simulate‑conversation helpers
├─ prep_responses.py        # Prep helper for evaluation
├─ rag.py                   # Simple RAG indexing/query helpers
├─ requirements.txt         # Python dependencies
├─ Dockerfile               # Optional container build
├─ .gitattributes
└─ README.md                # You are here :)
```

---

## Quickstart

### 1) Python setup

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Environment variables

Create a `.env` in the project root (same folder as `fresh_app_v2.py`). Minimal example:

```dotenv
# Used by the Well‑being Planner (CrewAI) tab
open_ai_key_for_crew_ai=sk-...

# Optional TTS configuration for the guided meditation
# TTS_PROVIDER=gtts               # or: edge | coqui | elevenlabs
# ELEVEN_API_KEY=...              # if using ElevenLabs
# EDGE_VOICE=en-US-JennyNeural    # if using edge-tts
# COQUI_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
```

> Some tabs may allow choosing models/keys in the UI.  
> The **Planner** currently works with the key above (and/or an in‑tab field if present in your build).

### 3) Run the app

```bash
streamlit run fresh_app_v2.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## Using the App

### Simulate a Conversation 🧪🤖
1. In that tab, paste a **Client Profile** (e.g., `Age 24 student; recently moved... sleep irregular...`).
2. Click **Generate Synthetic Dialogue** to produce a multi‑turn conversation.
3. Optionally **Download Transcript**.

### Well‑being Planner 📅🧘
1. Ensure your `.env` has `open_ai_key_for_crew_ai=sk-...` (or paste a key in the tab if the field is available).
2. Upload one **.txt** transcript (client–therapist chat).
3. Click **Create Plan & Meditation**.
4. The app displays:
   - **Transcript Summary**
   - **7‑Day Well‑being Plan** (Markdown, Day 1 … Day 7)
   - **Meditation Script** and an **MP3** player  
     (audio saved locally as `guided_meditation.mp3`)

### RAG (optional)
- Place your files into `rag_data/`.
- Build/update the index (if needed):

  ```bash
  python rag.py
  ```

- Use the app’s RAG controls to query your docs (index artifacts stored in `index_storage/`).

### Evaluation (optional)
- Use `prep_responses.py` to format generations and `judge.py` to score them.
- Outputs/examples are kept under `Results/`.

---

## Streamlit Configuration

- `config.toml` sets app defaults (e.g., dark mode). Example:

```toml
[theme]
base = "dark"
```

Adjust as needed per Streamlit docs.

---

## Docker (optional)

```bash
# Build
docker build -t tellme-assistant .

# Run (exposes Streamlit on 8501)
docker run --rm -p 8501:8501 --env-file .env tellme-assistant
```

---

## Troubleshooting

- **AuthenticationError / “You didn’t provide an API key.”**  
  Ensure `.env` includes `open_ai_key_for_crew_ai=sk-...` (or provide the key in‑tab if available) and **restart** Streamlit so the new env is loaded.

- **Only meditation shows but not the plan**  
  Update to the latest `crew_ai.py` that collects and returns **summary**, **plan**, and **meditation**, and ensure the tab renders all three fields.

- **TTS provider errors**  
  Install the provider’s dependency (`pip install edge-tts`, `pip install TTS`, `pip install elevenlabs`) and set the related env vars.

- **Ollama (if used in other tabs)**  
  Start the daemon and pull a model:
  ```bash
  ollama serve
  ollama pull llama3.1:8b-instruct
  ```

---

## Tech Stack

- **UI:** Streamlit  
- **LLMs:** OpenAI (planner), plus optional Anthropic/Ollama in other tabs  
- **Agents:** CrewAI (via LiteLLM under the hood)  
- **RAG:** Simple local index (`rag.py`, `index_storage/`)  
- **TTS:** gTTS / Edge‑TTS / Coqui TTS / ElevenLabs (configurable)

---

## Roadmap

- In‑tab API key entry for the CrewAI planner (UI‑first flow)
- Configurable model/provider for planner
- Save generated plans/MP3s into `Results/` with timestamped filenames

---

## License

Add a license (e.g., MIT) in `LICENSE` if you plan to distribute.

---

## Acknowledgments

- Streamlit template seed  
- CrewAI & LiteLLM ecosystem  
- TTS libraries: gTTS, Edge‑TTS, Coqui TTS, ElevenLabs
