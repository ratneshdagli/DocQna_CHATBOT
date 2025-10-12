import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
import tempfile

st.set_page_config(page_title="📚 Research Chatbot", layout="wide")
st.title("📚 Research Paper Chatbot with HyDE")

st.markdown("""
Upload a **research paper (PDF)** and start chatting with it!  
This version implements **HyDE** for better query understanding.
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

    # Store in FAISS
    vectorstore = FAISS.from_documents(splits, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Memory for conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Groq LLM for final answers
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant",temperature=0,max_tokens=50)

    # ---------------- HyDE Setup ----------------
    hyde_llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)
    hyde_prompt = ChatPromptTemplate.from_template(
        """Even if you do not know the answer, please generate a plausible answer to the following question.
Question: {question}
Answer:"""
    )
    hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()

    # ---------------- HyDE Retriever ----------------
    class HyDERetriever(BaseRetriever, BaseModel):
        """
        HyDE retriever: generates a hypothetical document using an LLM,
        then retrieves relevant documents from the base retriever.
        """
        base_retriever: BaseRetriever = Field(...)
        hyde_chain: any = Field(...)

        class Config:
            arbitrary_types_allowed = True

        def get_relevant_documents(self, query: str):
            # Generate hypothetical document
            hyde_doc_text = self.hyde_chain.invoke({"question": query})
            hyde_doc = Document(page_content=hyde_doc_text)
            # Retrieve relevant documents using the hypothetical doc as query
            return self.base_retriever.get_relevant_documents(hyde_doc.page_content)

    hyde_retriever = HyDERetriever(base_retriever=base_retriever, hyde_chain=hyde_chain)

    # Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hyde_retriever,
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
