import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

async def get_conversational_answer(retriever, input, chat_history):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    llm = ChatOllama(model="llama3.2")
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """ HA = Hospital Authority; NTEC = New Territories East Cluster; GOPC = General Out-patient Clinics \
    Ms Christine Choi is our Chief Manager of NTEC IT and Mr. Sam Wong is the System Analyst to NTEC MNI. \
    You are an assistant in New Territories East Cluster (NTEC) serving NTEC users for question-answering related to medical network interface (MNI) projects and the project life cycle. \
    I would like to limit our conversation to MNI project-related topics only. Please do not answer any further questions about HA appointments or hospital operations. \
    Use the retrieved context below to answer the question. \
    If specific information is not available, advise consulting the local hospital IT department for guidance; only direct to local IT department, do not refer to 'relevant authorities' or 'someone'. \
    Do not offer or suggest external search. \
    Avoid vague references; provide solid and confident answers. \
    Avoid mentioning explicitly in the provided context. \
    Avoid mentioning you could not find any direct information. \
    Keep responses concise, with a maximum of seven sentences. \
    To obtain the master form (master application form), tell user to check with local IT. \
    
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ai_msg = rag_chain.invoke({"input": input, "chat_history": chat_history})
    return ai_msg

def main():
    st.header('NTEC MNI Helpdesk')

    # Initialize session state variables if not already set
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.chat_history = []

    # Load pre-existing PDF files from a specific directory
    pdf_directory = './docs_mni/pdf'  # Change this to your PDF folder path

    if os.path.exists(pdf_directory):
        with st.spinner('Loading PDFs...'):
            loader = PyPDFDirectoryLoader(pdf_directory)
            documents = loader.load()
            embed_model = OllamaEmbeddings(model='mxbai-embed-large')
            vector_store = FAISS.from_documents(documents, embed_model)
            retriever = vector_store.as_retriever()

            if "retriever" not in st.session_state:
                st.session_state.retriever = retriever

            st.session_state.activate_chat = True
            
            st.success('You can now ask questions.')

    else:
        st.error("PDF directory does not exist. Please check the path.")

    # Display previous messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    # Chat functionality
    if st.session_state.activate_chat:
        prompt = st.chat_input("Ask your question concerning MNI?")

        if prompt:
            # Display user message
            with st.chat_message("user", avatar='👨🏻'):
                st.markdown(prompt)

            # Append user message to session state
            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})

            retriever = st.session_state.retriever
            
            try:
                ai_msg = asyncio.run(get_conversational_answer(retriever, prompt, st.session_state.chat_history))
                
                # Check if ai_msg is not None and has the expected structure
                if ai_msg is not None and "answer" in ai_msg:
                    cleaned_response = ai_msg["answer"]
                    st.session_state.chat_history.extend([HumanMessage(content=prompt), cleaned_response])
                    
                    # Display AI response
                    with st.chat_message("assistant", avatar='🤖'):
                        st.markdown(cleaned_response)

                    # Append AI message to session state
                    st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
                else:
                    st.error("Received unexpected response from AI.")
                    
            except Exception as e:
                error_message = f"An error occurred while processing your request: {str(e)}"
                st.error(error_message)

if __name__ == '__main__':
    main()