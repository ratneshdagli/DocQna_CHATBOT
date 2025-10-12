import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever


from langchain.chains.query_constructor.base import AttributeInfo
import tempfile

st.set_page_config(page_title="📚 Research Chatbot", layout="wide")
st.title("📚 Research Paper Chatbot with Metadata Filtering")

st.markdown("""
Upload a **research paper (PDF)** and start chatting with it!  
This version filters documents by metadata for more authoritative answers.
""")

# Sidebar for API Key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to start.")
    st.stop()

# Upload PDF
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

    # Add metadata to documents
    docs_with_metadata = []
    for split in splits:
        new_doc = Document(
            page_content=split.page_content,
            metadata={
                "publication_year": 2017,  # Example: Year for 'Attention Is All You Need'
                "author": "Vaswani et al."
            }
        )
        docs_with_metadata.append(new_doc)

    # Store in FAISS
    vectorstore = FAISS.from_documents(docs_with_metadata, embeddings)

    # Define metadata fields
    metadata_field_info = [
        AttributeInfo(
            name="publication_year",
            description="The year the paper was published",
            type="integer",
        ),
    ]
    document_content_description = "A research paper"

    # Initialize Self-Query Retriever
    retriever = SelfQueryRetriever.from_llm(
        llm=ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant"),
        vectorstore=vectorstore,
        document_content_description=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )

    # Memory for conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Groq LLM for answers
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    # Chat history state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous chat
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input
    if user_query := st.chat_input("Ask something about the paper..."):
        st.chat_message("user").write(user_query)
        result = qa_chain({"question": user_query})
        answer = result["answer"]

        # Save history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.chat_message("assistant").write(answer)

else:
    st.info("👆 Upload a research paper to start chatting.")
