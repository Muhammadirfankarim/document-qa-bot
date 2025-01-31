import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
# Update the import for text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Update the imports for embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import path
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import io
import requests
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

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
    """
    Manages the Gemini language model initialization and inference using Google's Generative AI.
    This simplified version focuses on direct integration with Google's API through LangChain.
    """
    
    def __init__(self):
        # Initialize model attribute as None - will be set during initialization
        self.model = None
        
    def initialize_model(self):
        """
        Initializes the Gemini model using environment variables for authentication.
        Includes proper error handling and user feedback through Streamlit.
        
        Returns:
            ChatGoogleGenerativeAI: Initialized model instance or None if initialization fails
        """
        try:
            # Only initialize if model hasn't been created yet
            if self.model is None:
                with st.spinner("Initializing Gemini... 🚀"):
                    
                    # Create the model instance with optimized parameters
                    self.model = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro",
                        temperature=0,     # Controls response creativity
                        #streaming=True      # Enable streaming responses
                    )
                    
                    # Show success message and celebratory animations
                    st.success("✨ Gemini model loaded successfully!")
                    st.balloons()
                    #st.snow()
            
            return self.model
            
        except Exception as e:
            # Provide detailed error feedback to the user
            st.error(f"Error initializing Gemini model: {str(e)}")
            return None

    def get_model(self):
        """
        Provides access to the initialized model instance.
        Ensures model is initialized before use.
        
        Returns:
            ChatGoogleGenerativeAI: The initialized model instance
        """
        return self.initialize_model()

class VectorStoreManager:
    """Manages vector store operations with proper error handling"""
    
    @staticmethod
    def check_vector_store_exists():
        return os.path.exists(VECTOR_STORE_PATH) and os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss")

    @staticmethod
    def safe_load_vector_store(embedding_model):
        try:
            if not VectorStoreManager.check_vector_store_exists():
                st.error("Please process documents first before asking questions.")
                return None
                
            return FAISS.load_local(
                VECTOR_STORE_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None

class DocumentProcessor:
    """Handles document processing and vector store creation"""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embedding_model"],
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG["chunk_size"],
            chunk_overlap=MODEL_CONFIG["chunk_overlap"]
        )

    def process_documents(self, files):
        try:
            text = self.extract_text(files)
            if not text:
                st.error("No valid text content found in the uploaded documents.")
                return False

            chunks = self.text_splitter.split_text(text)
            if not chunks:
                st.error("No valid text chunks created from the documents.")
                return False

            vector_store = None
            batch_size = 100
            
            # Create a more interactive progress bar
            progress_text = "Processing documents... Please wait"
            my_bar = st.progress(0, text=progress_text)
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, self.embedding_model)
                else:
                    temp_store = FAISS.from_texts(batch, self.embedding_model)
                    vector_store.merge_from(temp_store)
                
                progress = min(1.0, (i + batch_size) / len(chunks))
                my_bar.progress(progress, text=f"{progress_text} ({int(progress * 100)}%)")

            vector_store.save_local(VECTOR_STORE_PATH)
            st.success("🎉 Documents processed successfully!")
            st.balloons() # st.snow()
            return True

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False

    def extract_text(self, files):
        """Extract text from documents with error handling"""
        text_contents = []
        
        for file in files:
            try:
                text = None
                if file.name.endswith('.pdf'):
                    text = self._process_pdf(file)
                elif file.name.endswith('.txt'):
                    text = self._process_txt(file)
                elif file.name.endswith('.csv'):
                    text = self._process_csv(file)
                else:
                    st.warning(f"⚠️ Unsupported file type: {file.name}")
                    continue
                    
                if text and text.strip():
                    text_contents.append(text)
                    
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                
        return "\n\n".join(text_contents)

    def _process_pdf(self, file):
        try:
            reader = PdfReader(file)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return " ".join(text)
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def _process_txt(self, file):
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return ""

    def _process_csv(self, file):
        try:
            df = pd.read_csv(file)
            return df.to_string()
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return ""

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()

def load_image(local_path=None, github_url=None, default_path=None):
    """
    Loads an image from multiple possible sources with fallback options.
    
    Args:
        local_path (str): Full local path to image (e.g., 'D:/LLM-RAG/gemini_chatbot_qna/simple-qa-chatbot/Logo RSP (Color)')
        github_url (str): Raw GitHub URL to the image
        default_path (str): Relative path to check in current directory
    
    Returns:
        Image: The loaded image, or None if all loading attempts fail
    """
    def try_local_path(path):
        # Try common image extensions
        for ext in ['.png', '.jpg', '.jpeg', '.svg']:
            full_path = f"{path}{ext}"
            if os.path.exists(full_path):
                return full_path
        return None

    def try_github_url(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return io.BytesIO(response.content)
        except:
            return None
        return None

    # First, try the local path if provided
    if local_path:
        local_image = try_local_path(local_path)
        if local_image:
            return local_image

    # Then try GitHub URL if provided
    if github_url:
        github_image = try_github_url(github_url)
        if github_image:
            return github_image

    # Finally, try the default path in the current directory
    if default_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_image = try_local_path(os.path.join(current_dir, default_path))
        if default_image:
            return default_image

    return None

def create_sidebar():
    """Create and manage sidebar elements with flexible image loading"""
    with st.sidebar:
        # Configure image sources
        local_path = r"D:\LLM-RAG\gemini_chatbot_qna\simple-qa-chatbot\Logo.jpg"
        github_url = "https://raw.githubusercontent.com/Muhammadirfankarim/document-qa-bot/main/Logo.jpg"
        default_path = "Logo.jpg"

        # Load image using our flexible loader
        image_source = load_image(
            local_path=local_path,
            github_url=github_url,
            default_path=default_path
        )

        if image_source:
            st.image(image_source, width=300)
        else:
            st.error("Could not load the RSP logo")
        
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV",
            key="file_uploader"
        )

        col1, col2 = st.columns(2)
        with col1:
            clear_chat = st.button("🗑️ Clear Chat")
        with col2:
            clear_docs = st.button("🗑️ Clear Docs")

        if clear_chat:
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.balloons()
            
        if clear_docs:
            if os.path.exists(VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(VECTOR_STORE_PATH)
                st.success("Documents cleared!")
                st.snow()
                st.session_state.pop("vector_store_ready", None)

        return uploaded_files

def main():
    st.set_page_config(
        page_title="📚 Interactive Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f3e5f5;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()
    
    # Main title with emoji
    st.title("📚 Interactive Document Q&A")
    st.markdown("---")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    uploaded_files = create_sidebar()
    
    # Process documents if uploaded
    if uploaded_files:
        if st.sidebar.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                if doc_processor.process_documents(uploaded_files):
                    st.session_state["vector_store_ready"] = True
    
    # Display chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("💭 Ask about your documents..."):
        if not VectorStoreManager.check_vector_store_exists():
            st.error("📂 Please upload and process documents first!")
            return
            
        try:
            # Initialize model if needed
            llm = st.session_state.model_manager.initialize_model()
            if not llm:
                return
                
            # Process query
            vector_store = VectorStoreManager.safe_load_vector_store(
                doc_processor.embedding_model
            )
            
            if not vector_store:
                return
                
            # Set up QA chain with improved prompt
            qa_chain = load_qa_chain(
                llm,
                chain_type="stuff",
                prompt=PromptTemplate(
                    template="""Answer the question based on the context below. If the answer cannot be found in the context, respond with "I cannot find this information in the provided documents."

                    Context: {context}
                    
                    Question: {question}
                    
                    Please provide a clear and concise answer. If relevant, include specific quotes or references from the documents:
                    """,
                    input_variables=["context", "question"]
                )
            )
            
            # Get relevant documents and generate response
            docs = vector_store.similarity_search(query, k=3)
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = qa_chain(
                        {"input_documents": docs, "question": query},
                        return_only_outputs=True
                    )
                    st.markdown(response["output_text"])
            
            # Update chat history
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()