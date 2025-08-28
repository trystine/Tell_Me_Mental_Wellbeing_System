from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ChatPromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from datasets import Dataset
import pandas as pd
from datasets import Dataset, DatasetDict
import os
#from llama_index import VectorStoreIndex

def create_chat_engine(llm):

    #dataset1 = load_dataset("nbertagnolli/counsel-chat")

    #dataset2 = pd.read_json("data/combined_dataset.json") #load_dataset("Amod/mental_health_counseling_conversations")

    # df2 = pd.read_json(
    #     "data/combined_dataset.json", lines=True
    # )
    # dataset2 = DatasetDict({"train": df2})

    
    #train_ds = dataset1['train']

    #df = train_ds.to_pandas()

    # df= pd.read_csv('data/20220401_counsel_chat.csv')

    # df['upvotes'] = df['upvotes'].fillna(-1)

    # def select_best_answer(group):
    #     if (group['upvotes'] >= 0).any():
    #         return group.loc[group['upvotes'].idxmax()]
    #     else:
    #         return group.loc[group['views'].idxmax()]

    # best_rows = df.groupby('questionText', group_keys=False).apply(select_best_answer)
    # final_df = Dataset.from_pandas(best_rows.reset_index(drop=True))
    # dataset1 = DatasetDict({"train": final_df})


    # documents = []

    # for example in dataset1["train"]:
    #     context = example['questionText']
    #     response = example['answerText']
    #     content = f"Context: {context}\nResponse: {response}"
    #     documents.append(Document(text=content))

    # for example in dataset2['train']:
    #     if 'Context' in example and 'Response' in example:
    #         context = example['Context']
    #         response = example['Response']
    #         content = f"Context: {context}\nResponse: {response}"
    #         documents.append(Document(text=content))
    #     else:
    #         print("Skipping due to missing keys:", example)
        

    # documents= documents[:500]

    # splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)
    # Settings.llm= llm

    # Settings.embed_model = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-small-en-v1.5"
    # )

    print('Checking how many times index is created again')
    
    if os.path.exists("index_storage"):
        print('We are entering preloaded zone')
        splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)
        Settings.llm= llm

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir="index_storage")
        index = load_index_from_storage(storage_context, Settings=Settings, text_splitter=splitter)
        print('We successfully loaded the index')
    else:
        print("Creating and saving new index...")
        df2 = pd.read_json(
            "data/combined_dataset.json", lines=True
        )
        dataset2 = DatasetDict({"train": df2})

        df= pd.read_csv('data/20220401_counsel_chat.csv')

        df['upvotes'] = df['upvotes'].fillna(-1)

        def select_best_answer(group):
            if (group['upvotes'] >= 0).any():
                return group.loc[group['upvotes'].idxmax()]
            else:
                return group.loc[group['views'].idxmax()]

        best_rows = df.groupby('questionText', group_keys=False).apply(select_best_answer)
        final_df = Dataset.from_pandas(best_rows.reset_index(drop=True))
        dataset1 = DatasetDict({"train": final_df})


        documents = []

        for example in dataset1["train"]:
            context = example['questionText']
            response = example['answerText']
            content = f"Context: {context}\nResponse: {response}"
            documents.append(Document(text=content))

        for example in dataset2['train']:
            if 'Context' in example and 'Response' in example:
                context = example['Context']
                response = example['Response']
                content = f"Context: {context}\nResponse: {response}"
                documents.append(Document(text=content))
            else:
                print("Skipping due to missing keys:", example)
            

        documents= documents[:500]

        splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)
        Settings.llm= llm

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        index = VectorStoreIndex.from_documents(documents, Settings=Settings, text_splitter=splitter)
        #index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir="index_storage")

    print('We enter the retriever')
    retriever_name = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,     
    )
    print('We pass the retriever')
    simple_template= ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a compassionate therapist who listens and offers guidance, coping strategies, and emotional support. You help clients reflect on their emotions, offer comfort, and suggest healthy responses to stress, anxiety, or other mental health concerns."
            ),
        )])

    template = (
        "You are a compassionate therapist who actively listens, offers thoughtful guidance, and provides emotional support. "
        "Your role is to help clients reflect on their emotions, offer comfort, and suggest healthy coping strategies for stress, anxiety, and other mental health concerns.\n\n"
        
        "Below is the most relevant context from previous discussions that may guide your response:\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        
        "Using this information, along with the tone and intent of the conversation, please respond to the following query:\n"
        "{query_str}\n\n"
        
        "Ensure your response is natural, engaging, and human-like. Keep it concise yet thoughtful. "
        "Avoid repetition and strive to provide fresh insights while maintaining continuity in the conversation.\n\n"
        
        "If the user’s input is very short (e.g., 'Hi', 'Thank you', 'I see', 'Okay'), respond with a filler sentence and a follow-up question to keep the conversation flowing naturally.\n"
        
        "Above all, stay empathetic, relevant, and focused on the main topic."
    )

    qa_template= PromptTemplate(template)

    #lala_prompt =qa_template.format(context_str="You are a compassionate therapist who listens and offers guidance, coping strategies, and emotional support. You help clients reflect on their emotions, offer comfort, and suggest healthy responses to stress, anxiety, or other mental health concerns. \n")
    print('We enter the query engine')
    query_engine = RetrieverQueryEngine(retriever=retriever_name)
    print('We pass the query engine')
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_template}
    )

    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    old_prompt= (
        """
You are a highly empathetic and professional AI therapist. Your goal is to provide responses that are relevant, contextually appropriate, and aligned with the client’s situation. 

    You will receive a response from the Query Engine, but you must first verify that:
    1. It matches the context of the client’s current query and past responses.
    2. It is emotionally appropriate and supportive.
    3. It avoids irrelevant or unrelated information.

    If the Query Engine response is not fully aligned with the conversation, modify or rephrase it to make it contextually relevant and meaningful. If the Query Engine response is completely irrelevant, ignore it and generate a new response based on the chat history.

    Maintain a compassionate, professional, and non-judgmental tone. If the client shares a distressing experience, acknowledge their feelings before providing guidance.

    If you do not have sufficient information to provide an answer, ask an open-ended question to encourage the client to elaborate.

    Always prioritize clarity, warmth, and relevance in your responses.

"""
    )
    context_chat_engine_prompt=(
        """
    You are a compassionate and professional AI therapist providing empathetic, relevant support tailored to each client’s situation.

    If the user input is very short or simple (for example: “Hi”, “Hello”, “Thanks”, “Okay”), respond briefly and warmly with 1–2 short sentences only. Use natural, casual acknowledgments such as:
    - “Hi there! How can I support you today?”
    - “Thanks for reaching out. What’s on your mind?”

    Avoid detailed explanations or complex reflections for such messages.

    If the user input is longer and detailed, carefully review any retrieved information before replying:
    - Ensure your response directly addresses the client’s current query and conversation context.
    - Make sure it is emotionally sensitive and supportive.
    - Remove any irrelevant or off-topic content.

    Always maintain a warm, non-judgmental tone.

    If unsure how to respond, ask gentle, open-ended questions to encourage the client to share more.

    Prioritize brevity, clarity, empathy, and relevance to foster a supportive environment.

        """
    )

    chat_engine = ContextChatEngine.from_defaults(retriever=retriever_name, query_engine=query_engine, 
                                                memory=chat_memory, system_prompt=context_chat_engine_prompt)
    
    return chat_engine
        
