from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import re

import re

def _strip_meta(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)  # drop [notes]
    return text.strip()

def _normalize_label(text: str) -> str:
    t = re.sub(r"^\s*(Therapist|Counselor|Coach)\s*:", "Therapist:", text, flags=re.I)
    t = re.sub(r"^\s*(Client|User|Patient)\s*:", "Client:", t, flags=re.I)
    return t.strip()

def _ensure_prefixed(text: str, role: str) -> str:
    t = _normalize_label(_strip_meta(text))
    if not t.lower().startswith(f"{role.lower()}:"):
        t = f"{role}: {t}"
    # hard-correct if wrong:
    if role == "Therapist" and t.lower().startswith("client:"):
        t = "Therapist:" + t.split(":", 1)[1]
    if role == "Client" and t.lower().startswith("therapist:"):
        t = "Client:" + t.split(":", 1)[1]
    return t.strip()

def _body(line: str) -> str:
    # returns text after "Role:"
    return line.split(":", 1)[1].strip() if ":" in line else line.strip()

def simulate_conversation(simulated_therapist_conversation_chain,
                          simulated_client_conversation_chain):
    initial_response = "Hi, thank you for joining me today. How have you been adjusting lately?"
    chat_history = []

    # seed therapist opener
    ther_line = _ensure_prefixed(initial_response, "Therapist")
    chat_history.append(ther_line)

    # first client reply (feed only therapist body)
    raw_client = simulated_client_conversation_chain.predict(therapist_input=_body(ther_line))
    client_line = _ensure_prefixed(raw_client or "I'm feeling anxious and a bit isolated.", "Client")
    chat_history.append(client_line)

    # continue for 4 more lines (total ~6 lines as in your original)
    for _ in range(2):  # each loop adds Therapist + Client = 2 lines
        raw_ther = simulated_therapist_conversation_chain.predict(user_input=_body(client_line))
        ther_line = _ensure_prefixed(raw_ther or "That makes sense. What tends to trigger the worry most?", "Therapist")
        chat_history.append(ther_line)

        raw_client = simulated_client_conversation_chain.predict(therapist_input=_body(ther_line))
        client_line = _ensure_prefixed(raw_client or "Usually at night I start overthinking assignments.", "Client")
        chat_history.append(client_line)

    return chat_history


def Sentiment_chain(model):
    sentiment_prompt = PromptTemplate.from_template(
        template="""

        Analyze the sentiment of the {client_response} and classify it into one of the following categories:
        - **Suicidal**: Expresses intent to self-harm or suicide.
        - **Dangerous**: Expresses intent to harm others or suggests severe aggression.
        - **Highly Negative**: Deep sadness, hopelessness, severe frustration, or distress.
        - **Negative**: Mild sadness, irritation, disappointment.
        - **Neutral**: No strong emotion, general statement, or greetings.
        - **Positive**: Happy, optimistic, or encouraging statement.

        Input:
        Text: "{client_response}"

        Expected output: "The sentiment of the text is: *answer*"

        Note: I just want the output text to be of one line sentence as described in Expected output. No need to give reasoning

        """
    )
    
    sentiment_chain = LLMChain(
        llm=model, 
        prompt=sentiment_prompt
    )

    return sentiment_chain

def Therapist_LLM_Model(therapist_prompt,model):

    memory_live = ConversationBufferMemory(memory_key="history") 

    therapist_prompt_template = PromptTemplate(
        input_variables=["history", "user_input", "therapist_prompt"],
        template="""
        Use the template {therapist_prompt}

        \n\nConversation History:\n{history}\n\nClient: {user_input} \n\nTherapist:"""
    ).partial(therapist_prompt=therapist_prompt)
    
    # template="""
    #     You are a compassionate therapist who listens and offers guidance, coping strategies, and emotional support.
    #     You help clients reflect on their emotions, offer comfort, and suggest healthy responses to stress, anxiety, or other mental health concerns.

    #     \n\nConversation History:\n{history}\n\nClient: {user_input} \n\nTherapist:"""
    # def therapist_llm_conversation(user_input, history, memory_type, llama2):
    #     therapist_conversation = LLMChain(
    #         llm=llama2,
    #         prompt=therapist_prompt_template,
    #         memory=memory_type
    #     )
    #     response = therapist_conversation.run(
    #     history=history,
    #     user_input=user_input,
    #     )

    #     return response
    
    # user_input = "I feel overwhelmed with work and can't focus."
    # history = "Client: I have been feeling stressed lately.\nTherapist: Can you tell me more about what's causing the stress?"
    # memory_type = memory_type  # Define memory if needed
    # llama2 = llama2

    # response = therapist_llm_conversation(user_input, history, memory_type, llama2)
    # print("Therapist Response:", response)

    therapist_conversation = LLMChain(
        llm=model ,
        prompt=therapist_prompt_template,
        memory= memory_live#memory_type
    )

    return therapist_conversation

def Simulated_Client(client_prompt, model):

    memory_simulated = ConversationBufferMemory(memory_key="history") 
    client_template = PromptTemplate(
        input_variables=["history", "therapist_input", "client_prompt"],
        template="""
        Use the template {client_prompt}

        \n\nConversation History:\n{history}\n\nTherapist: {therapist_input}\nClient:"""
    ).partial(client_prompt=client_prompt)

    # client_template = PromptTemplate(
    #     input_variables=["history", "therapist_input"],
    #     template="""You are a client who is visiting a therapist for help with workplace anxiety and day-to-day stress.
    #     Respond authentically to the therapist's words.

    #     \n\nConversation History:\n{history}\n\nTherapist: {therapist_input}\nClient:"""
    # )
    llm_simulated_client_chain=LLMChain(
        llm=model , 
        prompt=client_template, 
        memory=memory_simulated#memory_type
        )

    return llm_simulated_client_chain

def create_client_prompt(model, client_profile):

    template = PromptTemplate(
        input_variables=["client_profile"],
        template="""
       SYSTEM:
        You are the CLIENT in a simulated therapy dialogue.

        - Only write responses as the CLIENT.
        - Do NOT write anything for the therapist.
        - Do NOT include explanations, notes, or meta-commentary.
        - Keep your replies natural, concise (1–4 sentences), and consistent with the persona below.
        - Always prefix your response with: "Client:"

        CLIENT PROFILE:
        {client_profile}
        """
    )
    

    client_prompt_model = LLMChain(
        llm=model ,
        prompt=template,
    )

    client_prompt= client_prompt_model.run(client_profile)

    return client_prompt

def create_therapist_prompt(model, client_profile):

    template = PromptTemplate(
        input_variables=["client_profile"],
        template="""
        SYSTEM:
        You are the THERAPIST in a simulated counseling conversation.

        - Only write responses as the THERAPIST.
        - Do NOT write anything for the client.
        - Do NOT include explanations, notes, or meta-commentary.
        - Use supportive, empathetic, and non-diagnostic language.
        - Keep responses concise (1–4 sentences).
        - Always prefix your response with: "Therapist:"

        CLIENT PROFILE (for context only — do not restate this to the client):
        {client_profile}
        """
    )
    

    therapist_prompt_model = LLMChain(
        llm=model ,
        prompt=template,
    )

    therapist_prompt= therapist_prompt_model.run(client_profile)

    return therapist_prompt


def rag_decider_chain(model):
    rag_decider_prompt = PromptTemplate.from_template("""
   You are a compassionate mental health AI therapist.

    Client message:
    \"\"\"{client_input}\"\"\"

    Retrieved context:
    \"\"\"{context_engine_response}\"\"\"

    Instruction:
    
    Analyze the {context_engine_response} if the response is relevant to the simple greeting or brief emotional check-in use it to provide your response.
                                                      
    Else respond warmly and empathetically without relying on the {context_engine_response}
                                                      
    If the {client_input} requires detailed guidance or factual info, use the retrieved {context_engine_response} to provide your response.
                                                      
    Provide a clear, empathetic, and contextually relevant reply.

        """
    )
    
    sentiment_chain = LLMChain(
        llm=model, 
        prompt=rag_decider_prompt
    )

    return sentiment_chain    




