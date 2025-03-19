from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from gtts import gTTS

load_dotenv()
openai_api_key = os.getenv('open_ai_key')

os.environ["OPENAI_API_KEY"] = openai_api_key

def task_agent_pipeline(chat_transcript):
    # chat_transcript = """
    #     Client: I've been feeling overwhelmed with work lately. It's like no matter how much I do, there's always more.
    #     Therapist: That sounds exhausting. Have you had time to relax or do something enjoyable?
    #     Client: Not really. I used to go on walks, but I haven't had the energy lately.
    #     Therapist: Lack of motivation can be tough. What usually helps you feel recharged?
    #     Client: Listening to music and journaling used to help, but I stopped doing that too...
    #     """

    print("Reached to crew ai")
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
        print(os.getcwd())
        print('result')
        print(result)
        print(type(result))
        tts = gTTS(text=str(result), lang="en", slow=True)
        tts.save("guided_meditation.mp3") 

        return "Guided meditation audio has been generated and saved as 'guided_meditation.mp3'."
    
    meditation_task = Task(
        description="Create a guided meditation script based on the user's emotional state and well-being plan. Avoid using characters like * in the script",
        agent=meditation_audio_agent,
        expected_output="A guided meditation script and an MP3 audio file.",
        context=[transcript_task, plan_task],
        callback=generate_meditation_audio 
    )

    wellness_crew = Crew(
        agents=[transcript_analysis_agent,  plan_generator_agent, meditation_audio_agent],
        tasks=[transcript_task, plan_task, meditation_task]
    )

    result = wellness_crew.kickoff(inputs={"user_input": chat_transcript})
    
    return result