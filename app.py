import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import torch
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Document, Settings
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
import llm_models as llm_models_file
import crew_ai as crew_ai_file    
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

st.set_page_config(page_title="Chatbot", page_icon=":speech_balloon:", layout="wide")

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

load_dotenv()
openai_api_key = os.getenv('open_ai_key')

os.environ["OPENAI_API_KEY"] = openai_api_key
torch.classes.__path__ = []

ss=st.session_state

llm_ollama = Ollama(model="llama2")

gpt_llm=ChatOpenAI(model="gpt-4o", temperature=0.7)

ss.llama2 = gpt_llm#llm_ollama

ss.memory_live = ConversationBufferMemory(memory_key="history")  
ss.memory_simulated = ConversationBufferMemory(memory_key="history") 

ss.live_therapist_conversation_chain = llm_models_file.Therapist_LLM_Model(ss.memory_live, ss.llama2)

ss.simulated_therapist_conversation_chain = llm_models_file.Therapist_LLM_Model(ss.memory_simulated, ss.llama2)

ss.simulated_client_conversation_chain =llm_models_file.Simulated_Client(ss.memory_simulated, ss.llama2)

ss.sentiment_chain= llm_models_file.Sentiment_chain(ss.llama2)

tab1, tab2, tab3 = st.tabs(["Chat with a Therapist", "Simulate a Conversation","Well-being Planner"])
if "selected_tab" not in ss:
    ss.selected_tab = "Tab 1"

with tab1:
    if ss.selected_tab=="Tab 1":
        
        st.title("Tell Me Chatbot")

        dataset = load_dataset("Amod/mental_health_counseling_conversations")

        documents = []
        for example in dataset['train']:  
            context = example['Context']  
            response = example['Response']
            content=f"Context: {context}\nResponse: {response}"
            doc = Document(text=content)
            documents.append(doc)

        documents= documents[0:100]
        
        Settings.llm=ss.llama2

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        index = VectorStoreIndex.from_documents(documents, Settings=Settings)

        retriever_name = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,
        )

        query_engine =RetrieverQueryEngine(retriever=retriever_name)

        if 'history' not in ss:
            ss.history = []

        for message in st.session_state.history:
            if message['role'] == 'user':
                st.markdown(f"<div style='text-align: right; padding: 5px; margin: 5px; background-color: #DCF8C6; border-radius: 10px; display: inline-block; color: black;'>{message['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; padding: 5px; margin: 5px; background-color: #E6E6E6; border-radius: 10px; display: inline-block; color: black;'>{message['message']}</div>", unsafe_allow_html=True)


        if "client_input" not in st.session_state:
            ss.client_input = ""
    
        client_input = st.text_area("Your message:", "", height=80)


        if st.button("Send"):
            if client_input:
                
                ss.history.append({'role': 'user', 'message': client_input})

                print('checking sentiment')
                sentiment_result = ss.sentiment_chain.run(client_response=client_input)
                
                if "suicidal" in sentiment_result or "dangerous" in sentiment_result:
                    response = "I'm really sorry you're feeling this way, but I cannot provide the help you need. Please reach out to a mental health professional or contact a crisis hotline immediately."
                    ss.history.append({'role': 'bot', 'message': response})
                    
                else:
                    with st.spinner('TellMe Bot is typing...'):
                        ai_response = query_engine.query(client_input)  #ss.live_therapist_conversation_chain.run(history=ss.memory_live, user_input=client_input, retrieved_knowledge=retrieved_knowledge_ans)
                        response = ai_response

                        ss.history.append({'role': 'bot', 'message': response})

                    st.rerun()
                
with tab2:
    if ss.selected_tab=="Tab 1":
        ss.chat_history= llm_models_file.simulate_conversation(ss.simulated_therapist_conversation_chain, ss.simulated_client_conversation_chain)
        for chat in ss.chat_history:
            st.write(chat)

with tab3:
    if ss.selected_tab=="Tab 1":
        print('reached to tab 3')
        result=crew_ai_file.task_agent_pipeline(ss.chat_history)
        st.write(result)

    




