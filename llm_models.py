from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def simulate_conversation(simulated_therapist_conversation_chain,simulated_client_conversation_chain):
   
    initial_response = "Hi thank you for joining me today. How are you doing today?"
    chat_history = []
    
    turn_count = 0  

    while turn_count < 5:  

        if turn_count == 0:
    
            chat_history.append(f"Tell Me ChatBot: {initial_response}")

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

def Sentiment_chain(llama2):
    sentiment_prompt = PromptTemplate.from_template(
        template="""
        Analyze the sentiment of the given text and classify it into one of the following categories:
        - **Suicidal**: Expresses intent to self-harm or suicide.
        - **Dangerous**: Expresses intent to harm others or suggests severe aggression.
        - **Highly Negative**: Deep sadness, hopelessness, severe frustration, or distress.
        - **Negative**: Mild sadness, irritation, disappointment.
        - **Neutral**: No strong emotion, general statement, or greetings.
        - **Positive**: Happy, optimistic, or encouraging statement.

        Input:
        Text: "{client_response}"

        Expected output: "The sentiment of the text is: ______"
        """
    )
    
    sentiment_chain = LLMChain(
        llm=llama2, 
        prompt=sentiment_prompt
    )

    return sentiment_chain

def Therapist_LLM_Model(memory_type, llama2):
    
    therapist_prompt_template = PromptTemplate(
        input_variables=["history", "user_input"],
        template="""
        You are a compassionate therapist who listens and offers guidance, coping strategies, and emotional support.
        You help clients reflect on their emotions, offer comfort, and suggest healthy responses to stress, anxiety, or other mental health concerns.

        \n\nConversation History:\n{history}\n\nClient: {user_input} \n\nTherapist:"""
    )
    
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
        llm=llama2 ,
        prompt=therapist_prompt_template,
        memory=memory_type
    )

    return therapist_conversation

def Simulated_Client(memory_type, llama2):

    client_template = PromptTemplate(
        input_variables=["history", "therapist_input"],
        template="""You are a client who is visiting a therapist for help with workplace anxiety and day-to-day stress.
        Respond authentically to the therapist's words.

        \n\nConversation History:\n{history}\n\nTherapist: {therapist_input}\nClient:"""
    )
    llm_simulated_client_chain=LLMChain(
        llm=llama2 , 
        prompt=client_template, 
        memory=memory_type
        )

    return llm_simulated_client_chain




