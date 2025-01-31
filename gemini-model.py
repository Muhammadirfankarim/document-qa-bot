import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
import os
import io
import requests
from PIL import Image

# Configuration
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vertex_embed_model": "text-multilingual-embedding-002",
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "llm_model": "gemini-1.5-pro"
}

VECTOR_STORE_PATH = "faiss_index"

class ModelManager:
    """Manages Gemini model initialization with streaming support."""
    
    def __init__(self):
        if "gemini_model" not in st.session_state:
            st.session_state.gemini_model = None

    def initialize_model(self):
        """Initializes the Gemini model with streaming enabled."""
        try:
            if st.session_state.gemini_model is None:
                api_key = st.secrets["GOOGLE_API_KEY"]
                if not api_key:
                    st.error("⚠️ Google API key is missing! Configure it in Streamlit secrets.")
                    return None
                
                with st.spinner("🚀 Initializing Gemini... Please wait."):
                    st.session_state.gemini_model = ChatGoogleGenerativeAI(
                        model=MODEL_CONFIG["llm_model"],
                        temperature=0,
                        streaming=True  # 🔥 Enable streaming responses
                    )
                    st.success("✨ Gemini model is ready!")
            return st.session_state.gemini_model
        except Exception as e:
            st.error(f"Error initializing Gemini model: {str(e)}")
            return None

class VectorStoreManager:
    """Manages vector store operations."""
    
    @staticmethod
    def check_vector_store_exists():
        return os.path.exists(VECTOR_STORE_PATH) and os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss")

    @staticmethod
    def safe_load_vector_store(embedding_model):
        try:
            if not VectorStoreManager.check_vector_store_exists():
                st.error("📂 Process documents before asking questions.")
                return None
                
            return FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None

class DocumentProcessor:
    """Handles document processing."""
    
    def __init__(self):
        self.embedding_model = VertexAIEmbeddings(
            model_name=MODEL_CONFIG["vertex_embed_model"]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG["chunk_size"],
            chunk_overlap=MODEL_CONFIG["chunk_overlap"]
        )

    def process_documents(self, files):
        try:
            text = self.extract_text(files)
            if not text:
                st.error("No valid text content found.")
                return False

            chunks = self.text_splitter.split_text(text)
            if not chunks:
                st.error("No valid text chunks created.")
                return False

            vector_store = FAISS.from_texts(chunks, self.embedding_model)
            vector_store.save_local(VECTOR_STORE_PATH)
            st.success("✅ Documents processed successfully!")
            st.balloons()
            return True

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False

    def extract_text(self, files):
        """Extract text from documents."""
        text_contents = []
        
        for file in files:
            try:
                if file.name.endswith('.pdf'):
                    text = self._process_pdf(file)
                elif file.name.endswith('.txt'):
                    text = self._process_txt(file)
                elif file.name.endswith('.csv'):
                    text = self._process_csv(file)
                else:
                    st.warning(f"⚠️ Unsupported file: {file.name}")
                    continue
                    
                if text.strip():
                    text_contents.append(text)
                    
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                
        return "\n\n".join(text_contents)

    def _process_pdf(self, file):
        try:
            reader = PdfReader(file)
            return " ".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def _process_txt(self, file):
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""

    def _process_csv(self, file):
        try:
            df = pd.read_csv(file)
            return df.to_string()
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return ""

def initialize_session_state():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()

def create_sidebar():
    """Sidebar UI."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/Muhammadirfankarim/document-qa-bot/main/Logo.jpg", width=300)
        st.header("📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDFs, TXTs, or CSVs",
            accept_multiple_files=True,
            key="file_uploader"
        )

        if st.button("🔄 Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    doc_processor = DocumentProcessor()
                    if doc_processor.process_documents(uploaded_files):
                        st.session_state["vector_store_ready"] = True

        st.button("🗑️ Clear Chat", on_click=lambda: st.session_state.update(messages=[]))
        return uploaded_files

def main():
    """Main Streamlit App."""
    st.set_page_config(page_title="📚 Document Q&A Chatbot", page_icon="📖", layout="wide")

    initialize_session_state()
    st.title("📚 Document Q&A Chatbot")
    
    create_sidebar()

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("💭 Ask about your documents..."):
        if not VectorStoreManager.check_vector_store_exists():
            st.error("📂 Please upload and process documents first!")
            return
            
        try:
            llm = st.session_state.model_manager.initialize_model()
            if not llm:
                return
                
            vector_store = VectorStoreManager.safe_load_vector_store(DocumentProcessor().embedding_model)
            if not vector_store:
                return
                
            qa_chain = load_qa_chain(
                llm,
                chain_type="stuff",
                prompt=PromptTemplate(
                    template="""Answer based on the provided documents. 
                    If no answer is found, say "I cannot find this information."

                    Context: {context}
                    Question: {question}
                    """,
                    input_variables=["context", "question"]
                )
            )
            
            docs = vector_store.similarity_search(query, k=3)
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                response_container = st.empty()
                response_stream = qa_chain.stream({"input_documents": docs, "question": query})

                full_response = ""
                for chunk in response_stream:
                    full_response += chunk["output_text"]
                    response_container.markdown(full_response)

                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
