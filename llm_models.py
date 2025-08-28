from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

def simulate_conversation(simulated_therapist_conversation_chain,simulated_client_conversation_chain):
   
    initial_response = "Hi thank you for joining me today. How are you doing today?"
    chat_history = []
    
    turn_count = 0  

    while turn_count < 5:  

        if turn_count == 0:
    
            chat_history.append(f"Therapist: {initial_response}")

            response_2 = simulated_client_conversation_chain.predict(therapist_input=initial_response)
        
            chat_history.append(f"Client: {response_2}")
            turn_count += 2 

        else:
        
            response_1 = simulated_therapist_conversation_chain.predict(user_input=response_2)
            chat_history.append(f"Therapist: {response_1}")

        
            response_2 = simulated_client_conversation_chain.predict(therapist_input=response_1)
            chat_history.append(f"Client: {response_2}")
            turn_count += 2  

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
        Based on the {client_profile} identify the key issues faced by the Client in a Client-Therapist scenario.
        Create a prompt that can be used as a template for an LLM who would be role-playing as this client.
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
        Based on the {client_profile} identify the key issues faced by the Client in a Client-Therapist scenario.
        Create a prompt that can be used as a template for an LLM who would be role-playing as this Therapist having a conversation with their client.
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




