from dotenv import load_dotenv
import os, asyncio, re
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# ---------- OPTIONAL TTS PROVIDERS ----------
# We'll try to import providers if available.
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    # pip install elevenlabs
    from elevenlabs import ElevenLabs
except Exception:
    ElevenLabs = None

try:
    # pip install edge-tts
    import edge_tts
except Exception:
    edge_tts = None

try:
    # pip install TTS
    from TTS.api import TTS
except Exception:
    TTS = None

load_dotenv()
openai_api_key = os.getenv('open_ai_key')
os.environ["OPENAI_API_KEY"] = openai_api_key

# ----------- TTS CONFIG -----------
TTS_PROVIDER   = os.getenv("TTS_PROVIDER", "gtts").lower()   # elevenlabs | edge | coqui | gtts
ELEVEN_KEY     = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE   = os.getenv("ELEVEN_VOICE", "Rachel")         # common friendly voices: Rachel, Bella
ELEVEN_MODEL   = os.getenv("ELEVEN_MODEL", "eleven_multilingual_v2")

EDGE_VOICE     = os.getenv("EDGE_VOICE", "en-US-JennyNeural")  # friendly neural voice
COQUI_MODEL    = os.getenv("COQUI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
OUTPUT_FILE    = "guided_meditation.mp3"

# ----------- TTS HELPERS -----------
def _clean_text(text: str) -> str:
    # keep it audio-friendly (remove Markdown symbols that TTS voices read awkwardly)
    text = re.sub(r"[*_`#>\[\](){}]", " ", str(text))
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def _chunk(text: str, max_chars: int = 4000):
    """Split long scripts into manageable chunks for TTS providers with limits."""
    text = text.strip()
    if len(text) <= max_chars:
        yield text
        return
    # split by sentences then pack
    parts = re.split(r"(?<=[.!?])\s+", text)
    cur = ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                yield cur
            cur = p
    if cur:
        yield cur

def tts_with_gtts(text: str, out_path: str = OUTPUT_FILE):
    if gTTS is None:
        raise RuntimeError("gTTS not installed. pip install gTTS")
    # gTTS "slow=True" sounds robotic; use normal speed
    audio = None
    from tempfile import NamedTemporaryFile
    # simple: write one file; gTTS has no streaming concat easily; for long text it's okay to keep single call
    tts = gTTS(text=_clean_text(text), lang="en", slow=False)
    tts.save(out_path)
    return out_path

def tts_with_elevenlabs(text: str, out_path: str = OUTPUT_FILE, voice: str = ELEVEN_VOICE, model: str = ELEVEN_MODEL):
    if ElevenLabs is None:
        raise RuntimeError("elevenlabs SDK not installed. pip install elevenlabs")
    if not ELEVEN_KEY:
        raise RuntimeError("Set ELEVEN_API_KEY in environment to use ElevenLabs TTS.")
    client = ElevenLabs(api_key=ELEVEN_KEY)

    # ElevenLabs supports streaming; we’ll write chunks sequentially
    with open(out_path, "wb") as f:
        for chunk in _chunk(_clean_text(text), max_chars=4800):
            audio_stream = client.generate(text=chunk, voice=voice, model=model)
            for b in audio_stream:
                f.write(b)
    return out_path

async def _edge_tts_async(text: str, out_path: str, voice: str):
    communicate = edge_tts.Communicate(text=_clean_text(text), voice=voice)
    await communicate.save(out_path)

def tts_with_edge(text: str, out_path: str = OUTPUT_FILE, voice: str = EDGE_VOICE):
    if edge_tts is None:
        raise RuntimeError("edge-tts not installed. pip install edge-tts")
    # edge-tts is async; run a simple event loop
    asyncio.run(_edge_tts_async(text, out_path, voice))
    return out_path

def tts_with_coqui(text: str, out_path: str = OUTPUT_FILE, model_name: str = COQUI_MODEL):
    if TTS is None:
        raise RuntimeError("Coqui TTS not installed. pip install TTS")
    # XTTS v2 is multilingual and has nice timbre; CPU works but slower on large texts.
    # You can provide `speaker_wav="voices/soothing.wav"` if you have a reference voice.
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    # If you hit speed/memory issues, chunk:
    chunks = list(_chunk(_clean_text(text), max_chars=800))
    if len(chunks) == 1:
        tts.tts_to_file(text=chunks[0], file_path=out_path, language="en")
    else:
        # stitch chunks into one file by writing temp WAVs then merge (simple approach: write each to mp3 and append)
        # For simplicity here we write/overwrite sequentially (last chunk wins). For perfect stitching, use pydub.
        try:
            from pydub import AudioSegment  # pip install pydub
            segs = []
            for i, ch in enumerate(chunks):
                tmp = f"__tmp_{i}.wav"
                tts.tts_to_file(text=ch, file_path=tmp, language="en")
                segs.append(AudioSegment.from_file(tmp))
                os.remove(tmp)
            final = sum(segs[1:], segs[0])
            final.export(out_path, format="mp3")
        except Exception:
            # fallback: single pass (last chunk only) to avoid errors if pydub not present
            tts.tts_to_file(text=chunks[-1], file_path=out_path, language="en")
    return out_path

def synthesize_tts(text: str, out_path: str = OUTPUT_FILE) -> str:
    provider = TTS_PROVIDER
    try:
        if provider == "elevenlabs":
            return tts_with_elevenlabs(text, out_path)
        elif provider == "edge":
            return tts_with_edge(text, out_path)
        elif provider == "coqui":
            return tts_with_coqui(text, out_path)
        elif provider == "gtts":
            return tts_with_gtts(text, out_path)
        else:
            # unknown → try ElevenLabs then edge then gTTS
            try:
                return tts_with_elevenlabs(text, out_path)
            except Exception:
                try:
                    return tts_with_edge(text, out_path)
                except Exception:
                    return tts_with_gtts(text, out_path)
    except Exception as e:
        # final fallback to gTTS if anything else fails
        if provider != "gtts":
            try:
                return tts_with_gTTS(text, out_path)
            except Exception:
                pass
        raise

# ----------- Your pipeline -----------
def task_agent_pipeline(chat_transcript):
    print("Reached crew_ai with transcript")
    print(chat_transcript)

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

    transcript_analysis_agent = Agent(
        name="Transcript Analyzer",
        role="Analyzes the client's chat with the AI therapist to extract emotions, key concerns, and sentiment trends.",
        goal="Extract user's emotional state and well-being indicators from the chat transcript.",
        backstory="An AI therapist assistant skilled in NLP-based sentiment and topic analysis.",
        llm=llm,
        verbose=True
    )

    plan_generator_agent = Agent(
        name="Plan Generator",
        role="Creates a personalized 1-week plan with activities, exercises, and affirmations.",
        goal="Generate a structured 7-day well-being improvement plan",
        backstory="An AI wellness coach that specializes in personalized mental health plans.",
        llm=llm,
        verbose=True
    )

    meditation_audio_agent = Agent(
        name="Meditation Generator",
        role="Creates a guided meditation script and generates an audio file for relaxation.",
        goal="Generate a calming meditation based on the user's emotional state and well-being plan.",
        backstory="An AI meditation coach that creates mindfulness and relaxation exercises.",
        llm=llm,
        verbose=True
    )

    transcript_task = Task(
        description="Analyze the chat transcript {user_input} and extract key emotions, concerns, and sentiment trends. Provide a Summary of the Transcript chat",
        agent=transcript_analysis_agent,
        expected_output="A summary of the client's emotional state, concerns, and well-being trends.",
    )

    plan_task = Task(
        description="Based on the Summary of the Transcript chat, generate a customized 1-week well-being plan with recommended exercises and different CBT Techniques.",
        agent=plan_generator_agent,
        expected_output="A structured 7-day well-being plan with personalized exercises and CBT techniques",
        context=[transcript_task]
    )

    def generate_meditation_audio(result):
        """
        Callback: takes the generated meditation script (result),
        cleans it, and synthesizes a single MP3 'guided_meditation.mp3'
        using the selected TTS provider.
        """
        script_text = _clean_text(str(result))
        path = synthesize_tts(script_text, out_path=OUTPUT_FILE)
        return f"Guided meditation audio generated: {path}  (provider={TTS_PROVIDER})"

    meditation_task = Task(
        description="Create a guided meditation script based on the user's emotional state and well-being plan. Avoid using characters like * in the script",
        agent=meditation_audio_agent,
        expected_output="A guided meditation script and an MP3 audio file.",
        context=[transcript_task, plan_task],
        callback=generate_meditation_audio
    )

    wellness_crew = Crew(
        agents=[transcript_analysis_agent, plan_generator_agent, meditation_audio_agent],
        tasks=[transcript_task, plan_task, meditation_task]
    )

    result = wellness_crew.kickoff(inputs={"user_input": chat_transcript})
    return result
