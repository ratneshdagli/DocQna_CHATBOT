import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="📚 Research Chatbot", layout="wide")
st.title("📚 Research Paper Chatbot")

st.markdown("""
Upload a **research paper (PDF)** and start chatting with it!  
Now powered by **Hybrid Retrieval (FAISS + BM25)** for better precision 🚀
""")

# ------------------ Sidebar ------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to start.")
    st.stop()

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ------------------ FAISS Retriever ------------------
    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ------------------ BM25 Retriever ------------------
    bm25_retriever = BM25Retriever.from_documents(splits)

    # ------------------ Ensemble Retriever (Hybrid) ------------------
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # Equal weighting for keyword + semantic
    )

    # ------------------ Conversation Memory ------------------
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # ------------------ LLM ------------------
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # ------------------ Conversational Chain ------------------
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory,
        verbose=True
    )

    # ------------------ Chat UI ------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous chat
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if user_query := st.chat_input("Ask something about the paper..."):
        st.chat_message("user").write(user_query)

        result = qa_chain({"question": user_query})
        answer = result["answer"]

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.chat_message("assistant").write(answer)

else:
    st.info("👆 Upload a research paper to start chatting.")
