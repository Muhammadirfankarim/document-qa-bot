# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
#from langchain_community.llms import Ollama
from langchain_ollama.chat_models import ChatOllama
import os

# Configuration
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 2000,
    "chunk_overlap": 200
}

VECTOR_STORE_PATH = "faiss_index"

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
            progress_bar = st.progress(0)
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, self.embedding_model)
                else:
                    temp_store = FAISS.from_texts(batch, self.embedding_model)
                    vector_store.merge_from(temp_store)
                
                progress = min(1.0, (i + batch_size) / len(chunks))
                progress_bar.progress(progress)

            vector_store.save_local(VECTOR_STORE_PATH)
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
                    st.warning(f"Unsupported file type: {file.name}")
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

def main():
    st.set_page_config(page_title="Document Q&A Bot", layout="wide")
    st.title("📚 Document Q&A Bot")
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    # File upload section
    with st.sidebar:
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV"
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    if doc_processor.process_documents(uploaded_files):
                        st.success("Documents processed successfully!")
                        st.session_state["vector_store_ready"] = True
                    else:
                        st.error("Failed to process documents. Please try again.")
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask about your documents..."):
        if not VectorStoreManager.check_vector_store_exists():
            st.error("Please upload and process documents first!")
            return
            
        try:
            # Process query
            vector_store = VectorStoreManager.safe_load_vector_store(
                doc_processor.embedding_model
            )
            
            if not vector_store:
                return
                
            # Initialize language model
            llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
            
            # Set up QA chain
            qa_chain = load_qa_chain(
                llm,
                chain_type="stuff",
                prompt=PromptTemplate(
                    template="""Answer based on the context below. If the answer isn't in the context, 
                    say "I cannot find this information in the provided documents."
                    
                    Context: {context}
                    Question: {question}
                    
                    Answer:""",
                    input_variables=["context", "question"]
                )
            )
            
            # Get relevant documents and generate response
            docs = vector_store.similarity_search(query, k=3)
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain(
                        {"input_documents": docs, "question": query},
                        return_only_outputs=True
                    )
                    st.write(response["output_text"])
            
            # Update chat history
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
