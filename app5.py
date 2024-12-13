import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage
import os
import asyncio
from dotenv import load_dotenv
import tempfile

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

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # File uploader
    uploaded_file = st.file_uploader("Upload additional PDF document", type=['pdf'], key="pdf_uploader")

    if uploaded_file is not None:
        with st.spinner('Processing uploaded file...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            new_documents = loader.load()

            if st.session_state.vector_store is None:
                embed_model = OllamaEmbeddings(model='mxbai-embed-large')
                st.session_state.vector_store = FAISS.from_documents(new_documents, embed_model)
            else:
                st.session_state.vector_store.add_documents(new_documents)

            st.session_state.retriever = st.session_state.vector_store.as_retriever()
            st.session_state.activate_chat = True
            st.success('File uploaded and processed successfully!')

    # Load pre-existing PDF files
    pdf_directory = './docs_mni/pdf'
    if os.path.exists(pdf_directory) and st.session_state.vector_store is None:
        with st.spinner('Loading PDFs...'):
            loader = PyPDFDirectoryLoader(pdf_directory)
            documents = loader.load()
            embed_model = OllamaEmbeddings(model='mxbai-embed-large')
            st.session_state.vector_store = FAISS.from_documents(documents, embed_model)
            st.session_state.retriever = st.session_state.vector_store.as_retriever()
            st.session_state.activate_chat = True
            st.success('You can now ask questions.')
    elif not os.path.exists(pdf_directory):
        st.warning("PDF directory does not exist. Please upload a file to start.")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    # Chat functionality
    if st.session_state.activate_chat:
        prompt = st.chat_input("Ask your question concerning MNI?")

        if prompt:
            with st.chat_message("user", avatar='👨🏻'):
                st.markdown(prompt)

            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})

            try:
                with st.spinner('Generating response...'):
                    ai_msg = asyncio.run(get_conversational_answer(st.session_state.retriever, prompt, st.session_state.chat_history))
                
                if ai_msg is not None and "answer" in ai_msg:
                    cleaned_response = ai_msg["answer"]
                    st.session_state.chat_history.extend([HumanMessage(content=prompt), cleaned_response])
                    
                    with st.chat_message("assistant", avatar='🤖'):
                        st.markdown(cleaned_response)

                    st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
                else:
                    st.error("Received unexpected response from AI.")
                    
            except Exception as e:
                st.error(f"An error occurred while processing your request: {str(e)}")

if __name__ == '__main__':
    main()
