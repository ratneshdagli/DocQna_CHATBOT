import streamlit as st
import tempfile
from typing import List
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="📚 Multi-Hop Research Chatbot", layout="wide")
st.title("📚 Multi-Hop RAG Chatbot with Groq (Llama 3.1)")

st.markdown("""
Upload a **research paper (PDF)** and ask **complex, multi-hop questions**.  
The chatbot breaks your query into sub-questions, retrieves answers for each, and synthesizes a final response.
""")

# Sidebar - API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API key to start.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📄 Upload a research paper (PDF)", type=["pdf"])

# -------------------- When PDF uploaded --------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Add metadata (optional)
    docs_with_metadata = [
        Document(
            page_content=split.page_content,
            metadata={
                "publication_year": 2017,
                "author": "Vaswani et al."
            }
        )
        for split in splits
    ]

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(docs_with_metadata, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # -------------------- Sub-question Decomposition --------------------
    sub_question_template = """
    You are a helpful assistant who generates stand-alone questions from a complex user query.
    Generate a JSON object containing a list of 2-3 sub-questions that need to be answered to address the user's query.
    The JSON object should have a single key, "questions", with a list of strings as the value.

    User Query: {query}
    """
    sub_question_prompt = ChatPromptTemplate.from_template(sub_question_template)
    sub_question_chain = sub_question_prompt | llm | JsonOutputParser()

    # -------------------- Multi-Hop Retrieval --------------------
    def retrieve_for_questions(questions: List[str]):
        """Retrieve documents for each sub-question."""
        retrieved = {}
        for q in questions:
            retrieved[q] = retriever.get_relevant_documents(q)
        return retrieved

    def format_retrieval_results(retrieved_docs: dict) -> str:
        """Format the retrieved documents for synthesis."""
        context = ""
        for question, docs in retrieved_docs.items():
            context += f"--- Context for: {question} ---\n"
            context += "\n".join([doc.page_content for doc in docs]) + "\n\n"
        return context

    final_answer_template = """
    Synthesize a comprehensive answer to the user's original query based on the provided context.
    Original Query: {original_query}
    Context:
    {context}
    """
    final_answer_prompt = ChatPromptTemplate.from_template(final_answer_template)

    # -------------------- Multi-hop Chain --------------------
    multi_hop_chain = (
        RunnablePassthrough.assign(
            sub_questions=sub_question_chain.pick("questions")
        )
        | RunnablePassthrough.assign(
            retrieved_docs=RunnableLambda(lambda x: retrieve_for_questions(x["sub_questions"]))
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_retrieval_results(x["retrieved_docs"]))
        )
        | final_answer_prompt
        | llm
        | StrOutputParser()
    )

    # -------------------- Chat UI --------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input("Ask a complex question (multi-hop supported)..."):
        st.chat_message("user").write(user_query)

        result = multi_hop_chain.invoke({"query": user_query, "original_query": user_query})
        answer = result

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.chat_message("assistant").write(answer)

else:
    st.info("👆 Upload a PDF to begin interacting.")
