import os
import sys
import io
import re
import time
import json
import base64
import random
import hashlib
from typing import Optional, List, Dict

from dotenv import load_dotenv
import streamlit as st

from llama_index.llms.langchain import LangChainLLM
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

try:
    from langchain_ollama.llms import OllamaLLM
    OLLAMA_OK = True
except Exception:
    OLLAMA_OK = False

try:
    from huggingface_hub import HfApi
    HFHUB_OK = True
except Exception:
    HFHUB_OK = False

import llm_models as llm_models_file
import rag as rag
import crew_ai as crew_ai_file

st.set_page_config(
    page_title="Tell Me — A Mental Well-Being Space",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # optional polish
        "About": "Tell Me is a safe space for individuals seeking some well-being advice or a self-reflection space. It also provides the research community to simulate some LLM generated client-therapist synthetic data. This is a research prototype, not medical device."
    },
)

if TORCH_OK:
    try:
        torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
    except Exception:
        pass

load_dotenv()

# ---------------------------------------------------------------------
# MODES: "public" (default) vs "study"
# - public: one chat; visible RAG toggle
# - study: access gate + two-part blinded flow (rag vs nonrag) with per-part ratings
# Switchable via sidebar and URL query (?mode=public|study) or env MODE
# ---------------------------------------------------------------------
DEFAULT_MODE = (os.getenv("MODE", "public") or "public").strip().lower()
qs = st.query_params 
mode_q = None
if isinstance(qs, dict) and "mode" in qs:
    mode_q = qs.get("mode")
    if isinstance(mode_q, list):
        mode_q = mode_q[0] if mode_q else None
if mode_q:
    DEFAULT_MODE = (mode_q or DEFAULT_MODE).strip().lower()

if "app_mode" not in st.session_state:
    st.session_state.app_mode = DEFAULT_MODE if DEFAULT_MODE in {"public", "study"} else "public"


# Access control (only used in study mode)
ACCESS_CODE = os.getenv("ACCESS_CODE", "")

# logging controls 
LOG_DATASET_REPO = os.getenv("LOG_DATASET_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_LOG_DIR = os.getenv("LOCAL_LOG_DIR", os.path.join("Results", "logs"))
ENABLE_LOGGING = bool(LOG_DATASET_REPO and HF_TOKEN and HFHUB_OK)
PREP_SPINNER_SECONDS = float(os.getenv("PREP_SPINNER_SECONDS", "1.2"))

# Stored Index Storage
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "/data/index_storage")

# Theme CSS
def inject_ui_css():
    st.markdown(
        """
    <style>
      :root{
        --text:#e7eaf0; --muted:#9aa4b2; --card-bg:#0f172a;
        --header-start:#0f1b2e; --header-end:#0b1324; --header-border:#1e2a44;
      }
      @media (prefers-color-scheme: light){
        :root{
          --text:#0f172a; --muted:#6b7280; --card-bg:#ffffff;
          --header-start:#f4f8ff; --header-end:#f9fcff; --header-border:#e6eefc;
        }
      }
      .stTabs [data-baseweb="tab"]{
        font-size:1.12rem;
        padding:10px 16px;
      }
      .stTabs [data-baseweb="tab"][aria-selected="true"]{
        font-weight:700;
        border-bottom:2px solid rgba(109,152,255,.6);
      }
      @media (max-width: 640px){
        .stTabs [data-baseweb="tab"]{ font-size:1.0rem; padding:8px 12px; }
      }
      .header-card{
        background:linear-gradient(135deg,var(--header-start) 0%,var(--header-end) 100%);
        border:1px solid var(--header-border);
        padding:18px 20px; border-radius:16px; margin-bottom:8px; color:var(--text);
        box-shadow:0 8px 30px rgba(0,0,0,.08);
      }
      .header-title{
        display:flex; flex-wrap:wrap; align-items:baseline; gap:.4rem;
        font-size:1.9rem; line-height:1.2; font-weight:800; margin:0; color:var(--text);
        letter-spacing:.2px;
      }
      
      .header-title .mini-pill{
        display:block;
        margin-top:10px;     
        margin-left:0;       
        width:fit-content;  
      }
      .header-sub{ color:var(--muted); margin-top:10px; font-size:.98rem; }
      .mini-pill{
        display:inline-block; padding:4px 10px; border-radius:999px;
        background:rgba(109,152,255,.12); border:1px solid rgba(109,152,255,.35);
        color:#8fb0ff; font-size:.78rem; margin-left:8px;
        backdrop-filter:saturate(140%) blur(6px);
      }

      .app-hero{
        background:linear-gradient(135deg,var(--header-start) 0%,var(--header-end) 100%);
        border:1px solid var(--header-border);
        border-radius:16px;
        padding:20px;
        position:relative; overflow:hidden;
        box-shadow:0 8px 30px rgba(0,0,0,.10);
      }
      .app-hero:after{
        content:""; position:absolute; right:-60px; top:-60px; width:180px; height:180px;
        background:radial-gradient(closest-side, rgba(109,152,255,.22), transparent 60%);
        filter:blur(10px);
      }
      .app-title{ display:flex; align-items:baseline; gap:.35rem; font-weight:800; font-size:1.75rem; color:var(--text); letter-spacing:.2px; }
      .app-title .mono{ font-weight:700; opacity:.9; font-size:1.1rem; }
      .app-meta{ margin-top:8px; display:flex; flex-wrap:wrap; gap:8px; }
      .badge{
        font-size:.72rem; padding:4px 10px; border-radius:999px; border:1px solid;
        backdrop-filter:saturate(140%) blur(6px);
      }
      .badge.study  { background:rgba(252,211,77,.12);  border-color:rgba(252,211,77,.35);  color:#fcd34d; }
      .badge.public { background:rgba(109,152,255,.12); border-color:rgba(109,152,255,.35); color:#8fb0ff; }
      .badge.neutral{ background:rgba(148,163,184,.12); border-color:rgba(148,163,184,.35); color:#a8b2c1; }
      .app-sub{ color:var(--muted); margin-top:8px; font-size:.98rem; }
      .hairline{ height:1px; margin:10px 0 6px; background:linear-gradient(to right, transparent, rgba(148,163,184,.35), transparent); }
      @media (max-width: 640px){
        .app-title{ font-size:1.45rem; }
        .app-sub{ font-size:.95rem; }
      }

      .card{
        background:var(--card-bg);
        border:1px solid rgba(237,240,247,.18);
        border-radius:12px; padding:12px 14px; margin-bottom:10px;
        box-shadow:0 1px 0 rgba(16,24,40,.02);
      }
      .stButton>button{ border-radius:10px; padding:8px 14px; }

      .block-container{ padding-top:.6rem !important; padding-bottom:.6rem !important; }
      header[data-testid="stHeader"]{ margin-bottom:0 !important; }
      .stAlert{ margin-top:6px !important; margin-bottom:8px !important; padding:10px 12px !important; }
      .header-card{ margin-top:0 !important; }

      .chat-input{ box-shadow:0 -6px 18px rgba(0,0,0,.04); }
    </style>
    """,
        unsafe_allow_html=True,
    )

def header_bar():
    mode = st.session_state.get("app_mode", "public")
    mode_label = "Research Prototype (Study)" if mode == "study" else "Public Preview"
    mode_class = "study" if mode == "study" else "public"

    st.markdown(
        f"""
      <div class="header-card">
        <div class="header-title">🌿 Tell Me — A Mental Well-Being Space</div>

        <div class="header-meta">
          <span class="mini-pill {mode_class}">{mode_label}</span>
          <span class="mini-pill neutral">Calm · Private · Supportive</span>
        </div>

        <div class="header-sub">
          Tell Me is a safe space for individuals seeking some well-being advice or a self-reflection space. It also provides the research community to simulate some LLM generated client-therapist synthetic data. This is a research prototype, not a medical device.
        </div>
      </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data()
def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_app_background(image_path: str):
    try:
        b64 = file_to_base64(image_path)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpeg;base64,{b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            [data-testid="stAppViewContainer"] {{ background-color: transparent; }}
            [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
            [data-testid="stToolbar"] {{ right: 0; }}
            </style>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        print("Background image failed:", e)

# Blindfolded ordering for 2 part study
def assign_order(pid: Optional[str]) -> List[str]:
    """Blinded condition order from a participant id.
    Returns ["rag","nonrag"] or ["nonrag","rag"].
    """
    force = os.getenv("FORCE_ORDER")
    if force in ("AB", "BA"):
        return ["rag", "nonrag"] if force == "AB" else ["nonrag", "rag"]
    if pid:
        h = int(hashlib.sha256(pid.encode()).hexdigest(), 16)
        return ["rag", "nonrag"] if (h % 2) == 0 else ["nonrag", "rag"]
    return random.choice([["rag", "nonrag"], ["nonrag", "rag"]])

_SANITIZERS = [
    (r"\s*\[\d+(?:,\s*\d+)*\]", ""),
    (r"\(?(?:Source|source)\s*:\s*[^)\n]+?\)?", ""),
    (r"https?://\S+", ""),
]

def sanitize(text: str) -> str:
    for pat, repl in _SANITIZERS:
        text = re.sub(pat, repl, text)
    return text.strip()


def as_text(x):
    if hasattr(x, "response"):
        return str(x.response)
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(x, BaseMessage):
            return x.content
    except Exception:
        pass
    if isinstance(x, dict):
        return x.get("text") or x.get("content") or str(x)
    return str(x)


# Optional HF API for study logging
_hf_api = HfApi() if ENABLE_LOGGING else None

def _write_local_log(row: dict):
    ts = row.get("ts", int(time.time()))
    date_dir = time.strftime("%Y-%m-%d", time.gmtime(ts))
    part = row.get("participant_id", "anon") or "anon"
    folder = os.path.join(LOCAL_LOG_DIR, date_dir)
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"{part}_{ts}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[local-log] wrote {fname}")

def _upload_log(row: dict):
    assert ENABLE_LOGGING and _hf_api, "Logging is disabled"
    try:
        _hf_api.create_repo(repo_id=LOG_DATASET_REPO, repo_type="dataset", private=True, token=HF_TOKEN)
    except Exception:
        pass
    date_dir = time.strftime("%Y-%m-%d", time.gmtime(row.get("ts", int(time.time()))))
    fname = f"logs/{date_dir}/{row.get('participant_id','anon')}_{row['ts']}.json"
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    _hf_api.upload_file(
        path_or_fileobj=io.BytesIO(payload),
        path_in_repo=fname,
        repo_id=LOG_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

def safe_upload_log(row: dict):
    if st.session_state.app_mode != "study":
        return
    _write_local_log(row)
    if ENABLE_LOGGING:
        _upload_log(row)

def reset_chat_state(reason: str = ""):
    """Clear all chat-related state so a new provider/key starts fresh."""
    ss = st.session_state
    # public
    ss.history = []
    ss.chat_input = ""
    # study
    for k in ["history_p1", "history_p2", "chat_input_p1", "chat_input_p2",
              "ratings_p1", "ratings_p2", "study_part_index", "study_order"]:
        ss.pop(k, None)
    ss.chat_engine_rag = None
    for k in ["sentiment_chain", "ai_usage_collected", "ai_usage"]:
        ss.pop(k, None)

def ensure_active_auth_signature(provider: str, key: Optional[str]) -> None:
    """If (provider, key) changed since last active model, reset the chat state.
    Stores a hash (not the raw key) in session_state["auth_sig_active"].
    """
    key = key or ""
    new_sig = hashlib.sha256(f"{provider}|{key}".encode("utf-8")).hexdigest()
    prev_sig = st.session_state.get("auth_sig_active")
    if prev_sig is None:
        st.session_state.auth_sig_active = new_sig
        return
    if new_sig != prev_sig:
        reset_chat_state("auth changed")
        st.session_state.auth_sig_active = new_sig
        try:
            st.toast("🔑 Provider/API key changed — chat reset.")
        except Exception:
            pass

def select_backend_and_model():
    """Sidebar controls for provider + API key. Returns a chat model.
    Manual key overrides env; when a *usable* (provider,key) pair changes,
    we reset the session state and start fresh.
    """
    with st.sidebar:
        st.markdown("### 🔧 Model Backend")
        provider = st.selectbox(
            "Choose a provider",
            [
                "OpenAI (GPT-4o)",
                "Anthropic (Claude 3.7 Sonnet)",
                *( ["Ollama (local)"] if OLLAMA_OK else [] ),
            ],
            help="Paste a key below for cloud providers. Ollama requires the runtime.",
        )

        typed_key = None
        effective_key = None

        if provider.startswith("OpenAI"):
            # Prefer manual entry; fallback to env
            typed_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...",
                                      help="Used only in your session; not logged.")
            effective_key = typed_key or os.getenv("OPENAI_API_KEY")
            if not effective_key:
                st.warning("Enter your OpenAI API key or set OPENAI_API_KEY.")
                return None
            # Detect change and reset before creating model
            ensure_active_auth_signature("openai", effective_key)
            if ChatOpenAI is None:
                st.error("LangChain OpenAI chat wrapper not available.")
                return None
            return ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=effective_key)

        if provider.startswith("Anthropic"):
            typed_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...",
                                      help="Used only in your session; not logged.")
            effective_key = typed_key or os.getenv("ANTHROPIC_API_KEY")
            if not effective_key:
                st.warning("Enter your Anthropic API key or set ANTHROPIC_API_KEY.")
                return None
            ensure_active_auth_signature("anthropic", effective_key)
            if ChatAnthropic is None:
                st.error("LangChain Anthropic chat wrapper not available.")
                return None
            return ChatAnthropic(model="claude-3-7-sonnet-latest", api_key=effective_key)

        if provider.startswith("Ollama"):
            if not OLLAMA_OK:
                st.error("Ollama is not available in this environment.")
                return None

            # Let the user choose which local Ollama model to run
            ollama_model_options = [
                "llama3",
                "mistral:7b",
                "gemma3",
                "phi4-mini:3.8b",
                "vitorcalvi/mentallama2:latest",
                "wmb/llamasupport",
                "ALIENTELLIGENCE/mentalwellness",
            ]
            selected_model = st.selectbox(
                "Ollama model",
                ollama_model_options,
                index=0,
                help="Choose which local Ollama model to use (must be installed with ollama).",
            )

            ensure_active_auth_signature("ollama", f"local|{selected_model}")
            base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            return OllamaLLM(model=selected_model, base_url=base_url)

        return None


def nonrag_reply(user_text: str, history: List[Dict[str, str]], model) -> str:
    style_prompt = (
        "You are a supportive, clear, non-clinical assistant. "
        "Answer in 4–6 sentences, be empathetic, avoid clinical claims."
    )
    hist_txt = "\n".join(f"{m['role'].capitalize()}: {m['message']}" for m in history[-4:])
    prompt = f"{style_prompt}\n\n{hist_txt}\nUser: {user_text}"
    out = model.invoke(prompt)
    return as_text(out)

@st.cache_resource(show_spinner=False)
def get_rag_engine(model_id: str):
    return rag.create_chat_engine(model_id)

# Set background image
set_app_background("bg.jpg")
inject_ui_css()

# Sidebar: mode toggle + basic settings
with st.sidebar:
    st.markdown("### 🌟 Mode")
    st.session_state.app_mode = st.radio(
        "Choose app mode",
        ["public", "study"],
        index=0 if st.session_state.app_mode == "public" else 1,
        help="Public removes consent + ratings; Study keeps them."
    )
    st.markdown("---")
    if st.session_state.app_mode == "public":
        rag_on = st.toggle("Use RAG retrieval", value=True, help="Turn off to see pure LLM responses.")
    else:
        # Hide the toggle in study mode
        rag_on = (os.getenv("RAG_IN_STUDY", "on").strip().lower() != "off")

header_bar()

st.info(
    "This is an educational prototype. It’s **not** medical/professional advice. "
    "If you need help, contact a professional or a local crisis line."
)
# Study-only gate (access code + consent)
if st.session_state.app_mode == "study":
    if ACCESS_CODE:
        st.write("This demo is access-restricted (study mode).")
        code = st.text_input("Enter access code to continue", type="password")
        if code.strip() != ACCESS_CODE:
            st.stop()

# Select backend + (optional) paste API key
model_obj = select_backend_and_model()

# Session state inits
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("participant_id", ss.get("participant_id", ""))
ss.setdefault("study_order", [])
ss.setdefault("study_part_index", None) 

# Calling the RAG based ChatEngine
if st.session_state.app_mode == "public" and rag_on and model_obj is not None and ss.get("chat_engine_rag") is None:
    wrapped = LangChainLLM(llm=model_obj)
    ss.chat_engine_rag = rag.create_chat_engine(wrapped)

def render_chat_messages(msgs: List[Dict[str, str]]):
    for message in msgs:
        if message['role'] == 'user':
            st.markdown(
                f"<div style='text-align:left;padding:8px;margin:5px;background-color:#DCF8C6;border-radius:12px;display:inline-block;max-width:80%;color:black;'>{message['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left;padding:8px;margin:5px;background-color:#E6E6E6;border-radius:12px;display:inline-block;max-width:80%;color:black;'>{message['message']}</div>",
                unsafe_allow_html=True,
            )

def render_part(part_idx: int, use_rag: bool, model_obj):
    part_name = "Part 1" if part_idx == 0 else "Part 2"
    hist_key = "history_p1" if part_idx == 0 else "history_p2"
    input_key = "chat_input_p1" if part_idx == 0 else "chat_input_p2"
    send_key = f"btn_send_p{part_idx+1}"
    clear_key = f"btn_clear_p{part_idx+1}"
    dl_key = f"btn_dl_p{part_idx+1}"

    st.subheader(f"{part_name} of 2")
    st.caption("Please chat naturally. When you're done, submit the quick ratings below to continue.")

    ss.setdefault(hist_key, [])
    history = ss[hist_key]

    render_chat_messages(history)

    user_msg = st.text_area("Your message…", key=input_key, height=100)
    can_send = bool(user_msg.strip()) and (model_obj is not None)
    send_clicked = st.button("Send", type="primary", disabled=not can_send, key=send_key)

    if st.button("🗑 Clear", key=clear_key):
        ss[hist_key] = []
        st.rerun()

    if send_clicked:
        history.append({"role": "user", "message": user_msg})

        # Build RAG engine on demand
        if use_rag and ss.get("chat_engine_rag") is None and model_obj is not None:
            wrapped = LangChainLLM(llm=model_obj)
            ss.chat_engine_rag = rag.create_chat_engine(wrapped)

        # Sentiment guard 
        if "sentiment_chain" not in ss and model_obj is not None:
            ss.sentiment_chain = llm_models_file.Sentiment_chain(model_obj)
        result = ss.sentiment_chain.invoke({"client_response": user_msg}) if model_obj else {"text": ""}
        last_sentiment = (result or {}).get("text", "—")

        if any(word in last_sentiment.lower() for word in ["suicidal", "dangerous"]):
            response = (
                "I'm really sorry you're feeling this way, but I cannot provide the help you need. "
                "Please reach out to a mental health professional or contact a crisis hotline immediately."
            )
        else:
            if use_rag and ss.get("chat_engine_rag") is not None:
                raw = ss.chat_engine_rag.chat(user_msg)
            else:
                raw = nonrag_reply(user_msg, history, model_obj)
            response = sanitize(as_text(raw))

        history.append({"role": "bot", "message": response})
        st.rerun()

    chat_text = "".join(
        f"User: {m['message']}\n\n" if m['role'] == "user" else f"Bot: {m['message']}\n\n" for m in history
    )
    st.download_button("📥 Download This Part", data=chat_text, file_name=f"tellme_{part_name.lower().replace(' ', '_')}.txt", mime="text/plain", key=dl_key)

    st.markdown("---")
    st.markdown(f"### Quick ratings for {part_name} (1 = Low, 5 = High)")
    metric_help = {
        "helpful": "How much this response helped you make progress on what you needed right now.",
        "supportive": "How caring, respectful, and non-judgmental the tone felt.",
        "clarity": "How easy it was to understand; clear, organized, free of jargon.",
        "grounded": "How well it stayed factual/relevant to your messages (no made-up details).",
        "overall": "Your overall impression of this chat in this part."
    }
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: helpful = st.slider("Helpfulness", 1, 5, 3, key=f"rate_helpful_p{part_idx+1}", help=metric_help["helpful"])
    with c2: supportive = st.slider("Supportive", 1, 5, 3, key=f"rate_supportive_p{part_idx+1}", help=metric_help["supportive"])
    with c3: clarity = st.slider("Clarity", 1, 5, 3, key=f"rate_clarity_p{part_idx+1}", help=metric_help["clarity"])
    with c4: grounded = st.slider("Groundedness", 1, 5, 3, key=f"rate_grounded_p{part_idx+1}", help=metric_help["grounded"])
    with c5: overall = st.slider("Overall", 1, 5, 3, key=f"rate_overall_p{part_idx+1}", help=metric_help["overall"])
    comments = st.text_area("Optional comments", key=f"rate_comments_p{part_idx+1}")

    save_label = "Save rating & Next → Part 2" if part_idx == 0 else "Save rating & Finish Study"
    if st.button(save_label, key=f"btn_save_rating_p{part_idx+1}"):
        ss[f"ratings_p{part_idx+1}"] = {
            "helpful": helpful,
            "supportive": supportive,
            "clarity": clarity,
            "grounded": grounded,
            "overall": overall,
            "comments": comments,
            "num_turns": sum(1 for m in history if m["role"] == "user"),
            "condition": "rag" if use_rag else "nonrag",
        }
        if part_idx == 0:
            ss.study_part_index = 1
            st.rerun()
        else:
            ss.study_part_index = 2  
            st.rerun()

def render_study_summary():
    st.success("Thank you! Both parts are complete.")
    st.markdown("---")
    row = {
        "ts": int(time.time()),
        "participant_id": st.session_state.get("participant_id",""),
        "order": st.session_state.get("study_order", []),
        "part1": st.session_state.get("ratings_p1", {}),
        "part2": st.session_state.get("ratings_p2", {}),
    }
    try:
        safe_upload_log(row)
        st.download_button(
            "⬇️ Download anonymized study record (JSON)",
            data=json.dumps(row, ensure_ascii=False, indent=2),
            file_name=f"tellme_study_{row['ts']}.json",
            mime="application/json",
            key="btn_dl_study_json_final",
        )
    except Exception as e:
        st.error(f"Logging failed: {e}")

tab_chat, tab_sim, tab_plan = st.tabs([
    "💬 Chat with an Assistant",
    "🧪 Simulate a Conversation",
    "📅 Well-being Planner",
])

with tab_chat:
    if st.session_state.app_mode == "public":
        st.title("Tell Me Assistant ✨💬")

        with st.expander("ℹ️ About the Tell Me assistant"):
            st.markdown("""
            **What it is**
            - Tell Me Assistant is a Mental Well-being Chatbot designed to help individuals process their thoughts and emotions in a supportive way.
            - It is not a substitute for professional care, but it offers a safe space for conversation and self-reflection.
            - The Assistant is created with care, recognizing that people may turn to it during moments of initial support. Its goal is to make such therapeutic-style interactions more accessible and approachable for everyone.

            **How it works**
            - Uses your selected **Model Backend** (sidebar). API keys are kept in your session.
            - Responses are short, clear, and empathetic. No diagnosis or medical advice.

            **Mode specifics**
            """)
            mode = st.session_state.get("app_mode", "public")
            if mode == "public":
                st.markdown("- **Public Preview**: You can toggle **Use RAG retrieval** in the sidebar for more grounded answers.")
            else:
                st.markdown("- **Research Prototype (Study)**: Retrieval settings are blinded; quick ratings appear after your chat.")

            st.markdown("""
            **How to use**
            1. Type what’s on your mind or what you need help with.
            2. Click **Send**. Use **Clear Chat** to start fresh; **Download Chat** saves a transcript.
            3. (Optional) Switch models in the sidebar; changing provider/key resets the chat to keep things clean.

            **Good things to try**
            """)
            st.code(
                "I’m feeling overwhelmed this week. Help me plan a gentle, 3-step routine and one 2-minute breathing exercise.",
                language="text"
            )
            st.code(
                "Give me three compassionate reframes for: “I’m behind and I’ll never catch up.”",
                language="text"
            )
            st.code(
                "I tend to ruminate at night. Can you suggest a short wind-down script I can read to myself?",
                language="text"
            )

            st.markdown("""
            **Safety & privacy**
            - This assistant can’t handle emergencies. If you’re in crisis, please contact local emergency services or a crisis hotline.
            - In **study mode**, anonymized ratings (and, if enabled by the host, logs) may be collected for research.
            """)

        render_chat_messages(ss.history)

        user_msg = st.text_area("Your message…", key="chat_input", height=100)

        can_send = bool(user_msg.strip()) and (model_obj is not None)

        send_clicked = st.button("Send", type="primary", disabled=not can_send, key="btn_send_public")
        if st.button("🗑 Clear Chat", key="btn_clear_public"):
            ss.history = []
            st.rerun()

        if send_clicked:
            ss.history.append({"role": "user", "message": user_msg})

            if "sentiment_chain" not in ss and model_obj is not None:
                ss.sentiment_chain = llm_models_file.Sentiment_chain(model_obj)
            result = ss.sentiment_chain.invoke({"client_response": user_msg}) if model_obj else {"text": ""}
            last_sentiment = (result or {}).get("text", "—")

            if any(word in last_sentiment.lower() for word in ["suicidal", "dangerous"]):
                response = (
                    "I'm really sorry you're feeling this way, but I cannot provide the help you need. "
                    "Please reach out to a mental health professional or contact a crisis hotline immediately."
                )
            else:
                if rag_on and ss.get("chat_engine_rag") is not None:
                    raw = ss.chat_engine_rag.chat(user_msg)
                else:
                    raw = nonrag_reply(user_msg, ss.history, model_obj)
                response = sanitize(as_text(raw))

            ss.history.append({"role": "bot", "message": response})
            st.rerun()

        # Download transcript
        chat_text = "".join(
            f"User: {m['message']}\n\n" if m['role'] == "user" else f"Bot: {m['message']}\n\n" for m in ss.history
        )
        st.download_button("📥 Download Chat", data=chat_text, file_name="tellme_chat.txt",
                           mime="text/plain", key="btn_dl_public")
    else:  # STUDY MODE — two-part blinded flow with per-part ratings
        st.title("Tell Me Assistant ✨💬")

        # Quick check-in (kept from original)
        if "ai_usage_collected" not in ss:
            st.markdown("#### Quick check-in before we start")
            with st.form("ai_usage_form", clear_on_submit=True):
                used_ai = st.radio(
                    "Have you ever used AI to process or reflect on your emotions?",
                    options=["Yes", "No", "Prefer not to say"], index=2,
                )
                details = st.text_input("If yes, which tools or how often? (optional)")
                proceed = st.form_submit_button("Continue")
            if proceed:
                ss.ai_usage_collected = True
                ss.ai_usage = {"used_ai_for_emotions": used_ai, "details": details.strip()}
                st.success("Thanks! You can begin now.")
                st.rerun()
            st.stop()

        # Participant code + blinded order
        if ss.study_part_index is None:
            with st.form("study_intro_form", clear_on_submit=True):
                st.write("To preserve anonymity, you may enter a **Participant Code** (optional). This only controls the order of the two chats.")
                pid = st.text_input("Participant Code (optional)", value=ss.get("participant_id", ""))
                start = st.form_submit_button("Start Part 1")
            if start:
                ss.participant_id = pid.strip()
                ss.study_order = assign_order(ss.participant_id)
                ss.study_part_index = 0
                ss.history_p1, ss.history_p2 = [], []
                st.rerun()
            st.stop()

        idx = ss.get("study_part_index")
        order = ss.get("study_order", [])

        if isinstance(idx, int) and idx >= 2:
            render_study_summary()
            st.stop()

        if not order:
            ss.study_order = assign_order(ss.get("participant_id",""))
            order = ss.study_order

        if idx is None:
            idx = 0
        else:
            idx = 0 if idx < 0 else 1 if idx > 1 else idx

        use_rag = (order[idx] == "rag")
        render_part(idx, use_rag, model_obj)


with tab_sim:
    st.title("Simulate a Conversation 🧪🤖")

    with st.expander("ℹ️ What is this tab?"):
        st.write(
            "This generates a **synthetic client–therapist conversation** from a short client profile. "
            "It helps create sample data for research and lets professionals inspect the dialogue quality. "
            "Outputs are created by an LLM and can guide future fine-tuning or evaluation."
        )
        st.markdown("**How to use**")
        st.markdown(
            "1) Write a brief persona in *Client Profile* (context, concerns, goals).\n"
            "2) Click **Send** to generate a multi-turn dialogue.\n"
            "3) Review the output and optionally **Download Transcript**."
        )
        st.markdown("**Example client profile**")
        st.code(
            "Age 24 student; recently moved cities. Feeling isolated and anxious about coursework. "
            "Sleep is irregular; tends to ruminate at night. Wants to build routines and reduce worry.",
            language="text",
        )

    client_profile = st.text_area(
        "Client Profile",
        key="simulate_chat",
        height=120,
        help="Describe the persona: context, concerns, coping, goals.",
    )

    gen_clicked = st.button("Generate Synthetic Dialogue", key="btn_sim_generate")
    if gen_clicked:
        if model_obj is None:
            st.error("Choose a backend and provide a key first.")
        else:
            # Build role-specific prompts
            client_prompt = llm_models_file.create_client_prompt(model_obj, client_profile)
            therapist_prompt = llm_models_file.create_therapist_prompt(model_obj, client_profile)

            chain_t = llm_models_file.Therapist_LLM_Model(therapist_prompt, model_obj)
            chain_c = llm_models_file.Simulated_Client(client_prompt, model_obj)

            # Run sim and render
            sim_hist = llm_models_file.simulate_conversation(chain_t, chain_c)
            for line in sim_hist:
                st.write(line)
            st.download_button(
                "📥 Download Transcript",
                data="\n\n".join(sim_hist),
                file_name="chat_history_simulator.txt",
                key="btn_sim_download",
            )


with tab_plan:
    st.title("Well-being Planner 📅🧘")

    with st.expander("ℹ️ What is this tab?"):
        st.write(
            "Upload a **client–therapist chat transcript (.txt)** and the agents (via CrewAI) will:\n"
            "- Analyze emotions & key concerns\n"
            "- Create a **7-day well-being plan** (e.g., CBT techniques, routines)\n"
            "- Generate a **guided meditation MP3** tailored to the transcript\n\n"
            "This is for research/education; it’s not medical advice."
        )
        st.markdown("**How to use**")
        st.markdown(
            "1) Paste your **OpenAI API key** below (used only for this run).\n"
            "2) Upload one **.txt** transcript (plain text).\n"
            "3) Click **Create Plan & Meditation**."
        )
        st.caption("Tip: Avoid personal identifiers in uploaded text.")

    crew_key = st.text_input("OpenAI API key (for this planner only)", type="password", key="crew_ai_openai_key")

    up = st.file_uploader(
        "Upload a .txt transcript",
        type=["txt"],
        key="planner_upload",
        help="Plain text only.",
    )

    plan_clicked = st.button(
        "Create Plan & Meditation",
        key="btn_plan_create",
        disabled=not (crew_key and up)
    )

    if plan_clicked:
        if not crew_key:
            st.error("Please provide your OpenAI API key.")
        elif not up:
            st.error("Please upload a .txt file first.")
        else:
            text_list = [line.strip() for line in up.read().decode("utf-8").split("\n") if line.strip()]

            result = crew_ai_file.task_agent_pipeline(
                text_list,
                openai_api_key=crew_key
            )

            st.subheader("📌 Transcript Summary")
            st.markdown(result.get("summary") or "_No summary returned._")

            st.subheader("📅 7-Day Well-being Plan")
            st.markdown(result.get("plan") or "_No plan returned._")

            st.subheader("🧘 Guided Meditation (Text)")
            st.markdown(result.get("meditation") or "_No meditation text returned._")

            try:
                with open("guided_meditation.mp3", "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")
            except FileNotFoundError:
                st.info("Meditation audio not found.")