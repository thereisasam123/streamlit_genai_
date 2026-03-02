import os
import streamlit as st


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


st.set_page_config(page_title="PDF RAG Chatbot", page_icon="")
st.title("Chat With Your PDF")

st.sidebar.header("Settings")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

if st.sidebar.button(" Clear Chat"):
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

@st.cache_resource
def build_qa(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="pdf_db"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF processed successfully!")

    if st.session_state.qa_chain is None:
        st.session_state.qa_chain = build_qa("temp.pdf")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask something about your PDF"):

    if not api_key:
        st.warning("Please enter your Groq API key first.")
        st.stop()

    if st.session_state.qa_chain is None:
        st.warning("Upload a PDF first.")
        st.stop()

    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain(prompt)

            answer = result["result"]
            sources = result["source_documents"]

            st.markdown(answer)

            with st.expander("Sources"):
                for doc in sources:
                    st.write(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
