import os
import sys
import io
import re
import time
import json
import base64
import random
import hashlib
from dotenv import load_dotenv
import streamlit as st
import torch
from langchain.chat_models import ChatOpenAI
import llm_models as llm_models_file
import rag as rag
import crew_ai as crew_ai_file
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from huggingface_hub import HfApi

st.set_page_config(page_title="Tell Me — A Mental Well Being Space", page_icon="🌿", layout="wide")

# Torch shim (from your code)
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# =========================
# ENV & Secrets
# =========================
load_dotenv()
openai_api_key = os.getenv('open_ai_key')
claude_api_key = os.getenv('claude_api_key')
os.environ["OPENAI_API_KEY"] = openai_api_key or ""
os.environ['ANTHROPIC_API_KEY'] = claude_api_key or ""

# Hugging Face logging config (set in Space → Settings → Secrets)
LOG_DATASET_REPO = os.getenv("LOG_DATASET_REPO")  # e.g. "your-username/tell_me_logs"
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional: force order during pilots: "AB" (rag→nonrag) or "BA"
FORCE_ORDER = os.getenv("FORCE_ORDER")  # "AB" | "BA" | None
# If you enable persistent storage on Spaces, point your index here
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "/data/index_storage")
# Local log directory (works on your laptop; on Spaces use /data/logs for persistence)
LOCAL_LOG_DIR = os.getenv("LOCAL_LOG_DIR", os.path.join("Results", "logs"))
ENABLE_LOGGING = bool(LOG_DATASET_REPO and HF_TOKEN)

# Neutral prep spinner duration (same for both arms)
PREP_SPINNER_SECONDS = float(os.getenv("PREP_SPINNER_SECONDS", "1.2"))

def inject_ui_css():
    st.markdown("""
    <style>
      :root{
        --text:#0f172a;          /* light text */
        --muted:#6b7280;
        --card-bg:#ffffff;
        --header-start:#f4f8ff;  /* header gradient (light) */
        --header-end:#f9fcff;
        --header-border:#e6eefc;
      }
      @media (prefers-color-scheme: dark){
        :root{
          --text:#e7eaf0;        /* dark text */
          --muted:#9aa4b2;
          --card-bg:#0f172a;
          --header-start:#0f1b2e; /* header gradient (dark) */
          --header-end:#0b1324;
          --header-border:#1e2a44;
        }
      }
      .header-card{
        background:linear-gradient(135deg,var(--header-start) 0%,var(--header-end) 100%);
        border:1px solid var(--header-border);
        padding:16px 20px;border-radius:14px;margin-bottom:8px;color:var(--text);
      }
      .header-title{font-size:1.25rem;font-weight:700;margin:0;color:var(--text);}
      .header-sub{color:var(--muted);margin-top:2px;}
      .mini-pill{
        display:inline-block;padding:2px 8px;border-radius:999px;
        background:rgba(109,152,255,.12);border:1px solid rgba(109,152,255,.3);
        color:#8fb0ff;font-size:12px;margin-left:8px;
      }
      .card{background:var(--card-bg);border:1px solid #edf0f7;border-radius:12px;
            padding:12px 14px;margin-bottom:10px;box-shadow:0 1px 0 rgba(16,24,40,.02);}
      .chat-input{box-shadow:0 -6px 18px rgba(0,0,0,.04);}
      .stButton>button{border-radius:10px;padding:8px 14px;}
    </style>
    """, unsafe_allow_html=True)

def header_bar():
    st.markdown("""
      <div class="header-card">
        <div class="header-title">🌿 Tell Me — A Mental Well Being Space <span class="mini-pill">Research Prototype</span></div>
        <div class="header-sub">A calm space to reflect. This assistant is supportive, not a substitute for professional care.</div>
      </div>
    """, unsafe_allow_html=True)

# Background helpers – makes BG visible in light/dark & "system" mode
@st.cache_data()
def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_app_background(image_path: str):
    """Apply bg to the Streamlit app container so it shows in light/dark & 'system' mode."""
    try:
        b64 = file_to_base64(image_path)
        st.markdown(f"""
            <style>
            .stApp {{
                background: url("data:image/jpeg;base64,{b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            [data-testid="stAppViewContainer"] {{ background-color: transparent; }}
            [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
            [data-testid="stToolbar"] {{ right: 0; }}
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        print("Background image failed:", e)

# =========================
# Blinding / Assignment helpers
# =========================
def assign_order(pid: str | None) -> list[str]:
    if FORCE_ORDER in ("AB", "BA"):
        return ["rag", "nonrag"] if FORCE_ORDER == "AB" else ["nonrag", "rag"]
    if pid:
        h = int(hashlib.sha256(pid.encode()).hexdigest(), 16)
        return ["rag", "nonrag"] if (h % 2) == 0 else ["nonrag", "rag"]
    return random.choice([["rag", "nonrag"], ["nonrag", "rag"]])

_SANITIZERS = [
    (r"\s*\[\d+(?:,\s*\d+)*\]", ""),                  # [1], [2,3]
    (r"\(?(?:Source|source)\s*:\s*[^)\n]+?\)?", ""),  # (Source: …)
    (r"https?://\S+", ""),                            # URLs
]
def sanitize(text: str) -> str:
    for pat, repl in _SANITIZERS:
        text = re.sub(pat, repl, text)
    return text.strip()

# =========================
# Logging helpers: local mirror + optional HF upload
# =========================
_hf_api = HfApi()

def write_local_log(row: dict):
    ts = row.get("ts", int(time.time()))
    date_dir = time.strftime("%Y-%m-%d", time.gmtime(ts))
    part = row.get("participant_id", "anon") or "anon"
    folder = os.path.join(LOCAL_LOG_DIR, date_dir)
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"{part}_{ts}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[local-log] wrote {fname}")

def upload_log(row: dict):
    assert LOG_DATASET_REPO and HF_TOKEN, "Set LOG_DATASET_REPO and HF_TOKEN in Space Secrets"
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
    write_local_log(row)           # always write locally
    if ENABLE_LOGGING:
        upload_log(row)            # upload to HF if configured


# =========================
# Non-RAG responder (direct model, no retrieval)
# =========================
def as_text(x):
    # LlamaIndex chat engine response
    if hasattr(x, "response"):
        return str(x.response)
    # LangChain message objects (OpenAI/Anthropic/etc.)
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(x, BaseMessage):
            return x.content
    except Exception:
        pass
    # Dicts or anything else
    if isinstance(x, dict):
        return x.get("text") or x.get("content") or str(x)
    return str(x)

def nonrag_reply(user_text: str, history: list[dict], model) -> str:
    style_prompt = (
        "You are a supportive, clear, non-clinical assistant. "
        "Answer in 4–6 sentences, be empathetic, avoid clinical claims."
    )
    hist_txt = "\n".join(f"{m['role'].capitalize()}: {m['message']}" for m in history[-4:])
    prompt = f"{style_prompt}\n\n{hist_txt}\nUser: {user_text}"
    out = model.invoke(prompt)
    return as_text(out)

# =========================
# Session state & model init
# =========================
ss = st.session_state

llm_llama = OllamaLLM(model="llama3")
llm_mistral = OllamaLLM(model="mistral:7b")
llm_gemma  = OllamaLLM(model="gemma3")
llm_claude = ChatAnthropic(model="claude-3-7-sonnet-latest")
llm_phi4   = OllamaLLM(model="phi4-mini:3.8b")
gpt_llm    = ChatOpenAI(model="gpt-4o", temperature=0.7)

llm_mentallama2   = OllamaLLM(model="vitorcalvi/mentallama2:latest")
llm_llamasupport  = OllamaLLM(model="wmb/llamasupport")
llm_al_luna       = OllamaLLM(model="ALIENTELLIGENCE/mentalwellness")

ss.model = llm_claude  # your chosen default

if "sentiment_chain" not in ss:
    ss.sentiment_chain = llm_models_file.Sentiment_chain(ss.model)
if "rag_decider" not in ss:
    ss.rag_decider = llm_models_file.rag_decider_chain(ss.model)
if "chat_engine_rag" not in ss:
    ss.chat_engine_rag = None

# =========================
# Apply background + header
# =========================
set_app_background('bg.jpg')
inject_ui_css()
header_bar()

# =========================
# Consent + assignment (two parts) — form hidden after submit
# =========================
st.info("This is a research prototype demo. It’s **not** medical/professional advice. If you need help, contact a professional or a local crisis line.")

if "started" not in ss:
    ss.started = False

# IMPORTANT: hide cache spinner and avoid hashing model by using _model
@st.cache_resource(show_spinner=False)
def get_rag_engine(_model, model_key: str = "default"):
    return rag.create_chat_engine(_model)

# Neutral prep used on first Send per part (looks the same for both arms)
def prepare_part(build_rag: bool):
    start = time.time()
    with st.spinner("Preparing your chat…"):
        if build_rag and ss.chat_engine_rag is None:
            model_key = getattr(ss.model, "model", getattr(ss.model, "model_name", "default"))
            ss.chat_engine_rag = get_rag_engine(ss.model, model_key=model_key)
        elapsed = time.time() - start
        pad = max(0.0, PREP_SPINNER_SECONDS - elapsed)
        if pad > 0:
            time.sleep(pad)

start_clicked = False
if not ss.get("started", False):
    with st.form("consent", clear_on_submit=True):
        st.caption("Please avoid sharing personal identifiers. You can stop any time by closing the app.")
        st.markdown(
            """**Consent (please read):**
- I am 18+ and consent to participate in this research demo.
- This is not medical advice and not for emergencies.
- I allow my responses (text + ratings) to be used for research and quality improvement.
- De-identification is used, but anonymity cannot be guaranteed; I will avoid personal identifiers.
- My messages may be processed by third-party providers (e.g., model/hosting services) to run the study.
- Participation is voluntary; I can stop at any time.
"""
        )
        consent = st.checkbox("I have read and agree to the above")
        participant_id = st.text_input("Participant ID (optional)")
        start_clicked = st.form_submit_button("Start")
    if start_clicked:
        if not consent:
            st.error("Please check the consent box to proceed.")
        else:
            ss.started = True
            ss.participant_id = (participant_id or "").strip()
            ss.arm_order = assign_order(ss.participant_id)   # ["rag","nonrag"] or ["nonrag","rag"]
            ss.phase = 0
            ss.arm = ss.arm_order[ss.phase]
            ss.block_logs = []
            ss.started_ts = int(time.time())
            ss.prepared = False  # will trigger neutral prep on first Send
            st.success("Thanks! You'll complete this short study in two parts. Please proceed with Part 1.")
            st.rerun()

if not ss.started:
    st.stop()

# =========================
# Sidebar (progress & privacy)
# =========================
with st.sidebar:
    st.markdown("### 🌿 Wellbeing Study")
    progress = 0.5 if getattr(ss, "phase", 0) == 1 else 0.25
    st.progress(progress, text=f"Part {getattr(ss,'phase',0)+1} of 2")
    pid = ss.get("participant_id") or "anonymous"
    st.caption(f"Participant: **{pid}**")
    with st.expander("🔒 About your data"):
        st.write(
            "- We log anonymized text you write and ratings you choose.\n"
            "- No names/emails are required. Keep PII out of messages.\n"
            "- You can download your transcript below the chat."
        )
    with st.expander("⚠️ Need help now?"):
        st.write(
            "If you’re in immediate distress, contact local emergency services or a suicide prevention hotline in your region."
        )

# =========================
# Streamlit fragments (safe fallback if not available)
# =========================
if hasattr(st, "fragment"):
    fragment = st.fragment
else:
    def fragment(func):
        return func

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Chat with a Therapist", "Simulate a Conversation", "Well-being Planner"])

@fragment
def render_chat_tab():
    st.title("Tell Me Chatbot ✨💬")
    st.caption(f"Part {ss.phase + 1} of 2")

    # One-time pre-question for Part 1: prior AI usage for emotions
    if ss.phase == 0 and not ss.get("ai_usage_collected", False):
        st.markdown("### Quick check-in before we start")
        with st.form("ai_usage_form", clear_on_submit=True):
            used_ai = st.radio(
                "Have you ever used AI to process or reflect on your emotions?",
                options=["Yes", "No", "Prefer not to say"],
                index=2,
                help="Examples: chatbots, journaling assistants, mental health apps, voice companions."
            )
            details = st.text_input(
                "If yes, which tools or how often? (optional)",
                help="You can share names (e.g., ChatGPT, Wysa) or frequency (e.g., weekly)."
            )
            proceed = st.form_submit_button("Continue")
        if proceed:
            ss.ai_usage_collected = True
            ss.ai_usage = {"used_ai_for_emotions": used_ai, "details": details.strip()}
            st.success("Thanks! You can begin Part 1 now.")
            st.rerun()
        st.stop()

    if 'history' not in ss:
        ss.history = []
    if 'reset_input' not in ss:
        ss.reset_input = False
    if 'prepared' not in ss:
        ss.prepared = False

    # Display chat history
    for message in ss.history:
        if message['role'] == 'user':
            st.markdown(
                f"<div style='text-align: left; padding: 8px; margin: 5px; background-color: #DCF8C6; "
                f"border-radius: 12px; display: inline-block; max-width: 80%; color: black;'>{message['message']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align: left; padding: 8px; margin: 5px; background-color: #E6E6E6; "
                f"border-radius: 12px; display: inline-block; max-width: 80%; color: black;'>{message['message']}</div>",
                unsafe_allow_html=True
            )

    # Sticky input CSS + container
    st.markdown(
        """
        <style>
        .chat-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: white;
            padding: 10px;
            border-top: 1px solid #ddd;
            z-index: 999;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)

    if ss.reset_input:
        st.session_state["chat_input"] = ""
        ss.reset_input = False

    client_input = st.text_area("Your message…", key="chat_input", height=80, label_visibility="collapsed")

    if st.button("Send"):
        if client_input.strip():
            # One-time neutral prep per part (builds RAG only if needed)
            if not ss.prepared:
                prepare_part(build_rag=(ss.arm == "rag"))
                ss.prepared = True

            ss.history.append({"role": "user", "message": client_input})

            # Sentiment guard
            sentiment_result = ss.sentiment_chain.invoke({"client_response": client_input})
            ss.last_sentiment = sentiment_result.get("text", "—")
            if any(word in ss.last_sentiment.lower() for word in ["suicidal", "dangerous"]):
                response = (
                    "I'm really sorry you're feeling this way, but I cannot provide the help you need. "
                    "Please reach out to a mental health professional or contact a crisis hotline immediately."
                )
            else:
                # Blinded branch: RAG vs non-RAG for THIS PART
                if ss.arm == "rag":
                    raw = ss.chat_engine_rag.chat(client_input)
                else:
                    raw = nonrag_reply(client_input, ss.history, ss.model)
                response = sanitize(as_text(raw))

            ss.history.append({"role": "bot", "message": response})
            ss.reset_input = True
            st.rerun()

    # Transcript download (this part only)
    chat_text = ""
    for msg in ss.history:
        role = "User" if msg['role'] == "user" else "Bot"
        chat_text += f"{role}: {msg['message']}\n\n"

    st.download_button(
        label="📥 Download",
        data=chat_text,
        file_name=f"chat_history_part{ss.phase+1}.txt",
        mime="text/plain"
    )

    if st.button("🗑 Clear Chat"):
        ss.history = []
        ss.reset_input = True
        st.rerun()

    # Ratings with tooltips
    st.markdown("---")
    st.markdown("### Quick ratings for this part (1 = Low, 5 = High)")
    metric_help = {
        "helpful": "How much this response helped you make progress on what you needed right now.",
        "supportive": "How caring, respectful, and non-judgmental the tone felt.",
        "clarity": "How easy it was to understand; clear, organized, free of jargon.",
        "grounded": "How well it stayed factual/relevant to your messages (no made-up details).",
        "overall": "Your overall impression of this chat in this part."
    }
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: helpful = st.slider("Helpfulness", 1, 5, 3, key=f"help_{ss.phase}", help=metric_help["helpful"])
    with c2: supportive = st.slider("Supportive", 1, 5, 3, key=f"sup_{ss.phase}", help=metric_help["supportive"])
    with c3: clarity = st.slider("Clarity", 1, 5, 3, key=f"clar_{ss.phase}", help=metric_help["clarity"])
    with c4: groundedness = st.slider("Groundedness", 1, 5, 3, key=f"gnd_{ss.phase}", help=metric_help["grounded"])
    with c5: overall = st.slider("Overall", 1, 5, 3, key=f"ovr_{ss.phase}", help=metric_help["overall"])

    with st.expander("ℹ️ What do these ratings mean?"):
        st.write(
            "- **Helpfulness**: helped you move forward.\n"
            "- **Supportive**: caring, respectful, non-judgmental tone.\n"
            "- **Clarity**: easy to understand.\n"
            "- **Groundedness**: factual & relevant to your messages.\n"
            "- **Overall**: your overall impression."
        )
    comments = st.text_area("Optional comments (this part only)", key=f"cmt_{ss.phase}")

    if st.button("Finish this part"):
        block = {
            "arm": ss.arm,
            "num_turns": sum(1 for m in ss.history if m["role"] == "user"),
            "helpful": helpful, "supportive": supportive, "clarity": clarity,
            "groundedness": groundedness, "overall": overall,
            "comments": comments,
            "turns": ss.history,
        }
        ss.block_logs.append(block)

        if ss.phase == 0:
            ss.phase = 1
            ss.arm = ss.arm_order[1]
            ss.history = []
            ss.reset_input = True
            ss.prepared = False  # reset so Part 2 shows neutral prep again
            st.success("Part 1 complete. Part 2 is ready. Continue when you’re ready.")
            st.rerun()
        else:
            try:
                row = {
                    "ts": int(time.time()),
                    "participant_id": ss.get("participant_id", ""),
                    "order": " -> ".join(ss.arm_order),
                    "ai_usage": ss.get("ai_usage", {}),  # include prior AI usage
                    "blocks": ss.block_logs,
                }
                safe_upload_log(row)  # local + optional HF upload
                try:
                    row = {
                        "ts": int(time.time()),
                        "participant_id": ss.get("participant_id", ""),
                        "order": " -> ".join(ss.arm_order),
                        "ai_usage": ss.get("ai_usage", {}),
                        "blocks": ss.block_logs,
                    }

                    # Optional local write if possible; no HF upload because secrets unset
                    safe_upload_log(row)

                    # ALWAYS offer user download (no storage costs)
                    json_payload = json.dumps(row, ensure_ascii=False, indent=2)
                    st.success("Thanks! Your feedback was recorded. You’ve completed both parts.")
                    st.download_button(
                        "⬇️ Download your anonymized study record (JSON)",
                        data=json_payload,
                        file_name=f"tellme_{ss.get('participant_id','anon') or 'anon'}_{row['ts']}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Logging failed: {e}")

                st.success("Thanks! Your feedback was recorded. You’ve completed both parts.")
            except Exception as e:
                st.error(f"Logging failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # end sticky input

@fragment
def render_simulation_tab():
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
        help="Describe the persona: context (work/school), main concerns, patterns/coping, and goals."
    )

    if st.button("Send", key='simulate'):
        client_prompt  = llm_models_file.create_client_prompt(ss.model, client_profile)
        therapist_prompt = llm_models_file.create_therapist_prompt(ss.model, client_profile)

        ss.simulated_therapist_conversation_chain = llm_models_file.Therapist_LLM_Model(client_prompt, ss.model)
        ss.simulated_client_conversation_chain   = llm_models_file.Simulated_Client(therapist_prompt, ss.model)

        ss.chat_history = llm_models_file.simulate_conversation(
            ss.simulated_therapist_conversation_chain,
            ss.simulated_client_conversation_chain
        )
        for chat in ss.chat_history:
            st.write(chat)

        chat_text = "\n\n".join(ss.chat_history)
        st.download_button(
            label="📥 Download Transcript",
            data=chat_text,
            file_name="chat_history_simulator.txt",
            mime="text/plain"
        )

@fragment
def render_planner_tab():
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
            "1) Upload one **.txt** transcript (plain text).\n"
            "2) Click **Send**.\n"
            "3) Review the plan and play/download the generated **guided_meditation.mp3**."
        )
        st.caption("Tip: Avoid personal identifiers in uploaded text.")

    uploaded_file = st.file_uploader(
        "Upload a .txt transcript",
        type=["txt"],
        help="One plain text file containing the client–therapist conversation. Avoid personal identifiers."
    )

    text_list = None
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        text_list = [line.strip() for line in content.split("\n") if line.strip()]

    if st.button("Send", key='planner'):
        if not text_list:
            st.error("Please upload a non-empty .txt file first.")
        else:
            result = crew_ai_file.task_agent_pipeline(text_list)
            st.write("📄 Well-being planner recommendation:")
            st.write(getattr(result, "raw", result))

            try:
                with open("guided_meditation.mp3", "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
            except FileNotFoundError:
                st.info("Meditation audio not found.")

# ---- Mount tabs ----
with tab1:
    render_chat_tab()

with tab2:
    render_simulation_tab()

with tab3:
    render_planner_tab()
