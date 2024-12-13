import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import os
import asyncio
from dotenv import load_dotenv
import json

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = """ HA = Hospital Authority; NTEC = New Territories East Cluster; GOPC = General Out-patient Clinics \
You are an assistant in New Territories East Cluster (NTEC) serving NTEC users for question-answering related to medical network interface (MNI) projects and the project life cycle. \
I would like to limit our conversation to MNI project-related topics only. Please do not answer any further questions about HA appointments or hospital operations. \
Use the retrieved context below to answer the question. \
If specific information is not available, advise consulting the local hospital IT department for guidance; only direct to local IT department, do not refer to 'relevant authorities' or 'someone'. \
Do not offer or suggest external search. \
Avoid vague references; provide solid and confident answers. \
Avoid mentioning explicitly in the provided context. \
Avoid mentioning you could not find any direct information. \
Keep responses concise, with a maximum of seven sentences. \
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

async def get_conversational_answer(retriever, input, chat_history):
    llm = ChatOllama(model="llama3.2")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    ai_msg = rag_chain.invoke({"input": input, "chat_history": chat_history})
    return ai_msg

def save_prompts():
    with open('prompts.json', 'w') as f:
        json.dump(st.session_state.prompts.to_dict('records'), f)

def load_prompts():
    try:
        with open('prompts.json', 'r') as f:
            return pd.DataFrame(json.load(f))
    except FileNotFoundError:
        return initialize_default_prompts()

def initialize_default_prompts():
    default_prompts = [
        {'Prompt': 'contextualize_q_system_prompt', 'Description': contextualize_q_system_prompt},
        {'Prompt': 'qa_system_prompt', 'Description': qa_system_prompt}
    ]
    return pd.DataFrame(default_prompts)

def prompt_management():
    st.title("Prompt Management")

    if 'prompts' not in st.session_state:
        st.session_state.prompts = load_prompts()

    prompt_container = st.container()
    with prompt_container:
        st.subheader("Existing Prompts")
        st.dataframe(st.session_state.prompts, use_container_width=True)

    st.subheader("Add New Prompt")
    new_prompt = st.text_input("Enter new prompt")
    new_description = st.text_area("Enter prompt description")

    if st.button("Add Prompt"):
        if new_prompt.strip() and new_description.strip():
            new_row = pd.DataFrame({'Prompt': [new_prompt], 'Description': [new_description]})
            st.session_state.prompts = pd.concat([st.session_state.prompts, new_row], ignore_index=True)
            save_prompts()
            st.success("Prompt added successfully!")
            prompt_container.empty()
            with prompt_container:
                st.subheader("Existing Prompts")
                st.dataframe(st.session_state.prompts, use_container_width=True)
        else:
            st.warning("Please enter both prompt and description.")

    st.subheader("Edit or Remove Prompts")
    if not st.session_state.prompts.empty:
        selected_prompt = st.selectbox("Select a prompt to edit or remove", st.session_state.prompts['Prompt'])
        selected_index = st.session_state.prompts[st.session_state.prompts['Prompt'] == selected_prompt].index[0]
        edit_prompt = st.text_area("Edit prompt", value=selected_prompt, height=150)
        edit_description = st.text_area("Edit description", value=st.session_state.prompts.loc[selected_index, 'Description'], height=300)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Update Prompt"):
                st.session_state.prompts.loc[selected_index] = [edit_prompt, edit_description]
                save_prompts()
                st.success("Prompt updated successfully!")
                prompt_container.empty()
                with prompt_container:
                    st.subheader("Existing Prompts")
                    st.dataframe(st.session_state.prompts, use_container_width=True)
        with col2:
            if st.button("Remove Prompt"):
                st.session_state.prompts = st.session_state.prompts.drop(selected_index).reset_index(drop=True)
                save_prompts()
                st.success("Prompt removed successfully!")
                prompt_container.empty()
                with prompt_container:
                    st.subheader("Existing Prompts")
                    st.dataframe(st.session_state.prompts, use_container_width=True)
        with col3:
            if st.button("Reset to Default Prompts"):
                st.session_state.prompts = initialize_default_prompts()
                save_prompts()
                st.success("Prompts reset to default!")
                prompt_container.empty()
                with prompt_container:
                    st.subheader("Existing Prompts")
                    st.dataframe(st.session_state.prompts, use_container_width=True)

def main():
    st.header('NTEC MNI Helpdesk')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
    if 'prompts' not in st.session_state:
        st.session_state.prompts = load_prompts()

    pdf_directory = './docs_mni/pdf'
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
            st.success('PDF files loaded successfully! You can now ask questions.')
    else:
        st.error("PDF directory does not exist. Please check the path.")

    tab1, tab2 = st.tabs(["Chat", "Prompt Management"])

    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message['avatar']):
                st.markdown(message["content"])

        if st.session_state.activate_chat:
            prompt = st.chat_input("Ask your question concerning MNI?")
            if prompt:
                with st.chat_message("user", avatar='👨🏻'):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})

                retriever = st.session_state.retriever
                try:
                    ai_msg = asyncio.run(get_conversational_answer(retriever, prompt, st.session_state.chat_history))
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

    with tab2:
        prompt_management()

if __name__ == '__main__':
    main()
