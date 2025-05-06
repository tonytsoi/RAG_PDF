import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

# Define constants
CHROMA_DB_DIRECTORY = "chroma_db"  # Directory where the Chroma database is stored
OLLAMA_MODEL = "llama3.2"  # Name of the Ollama model to use

# Ensure the Chroma DB directory exists
if not os.path.exists(CHROMA_DB_DIRECTORY):
    os.makedirs(CHROMA_DB_DIRECTORY)

# Function to initialize the Chroma database
@st.cache_resource
def initialize_chroma():
    os.remove(CHROMA_DB_DIRECTORY + '/chroma.sqlite3')
    embeddings = FastEmbedEmbeddings()
    return Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)

# Function to process and add documents to the Chroma database
def process_and_add_documents(uploaded_files, chroma_db):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Split into manageable chunks
    embeddings = FastEmbedEmbeddings()

    for uploaded_file in uploaded_files:
        try:
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
            # Load PDF and extract text
            loader = PyPDFLoader(temp_file)
            documents = loader.load()
            
            # Split documents into smaller chunks
            chunks = text_splitter.split_documents(documents)
            
            # Add chunks to the Chroma database
            chroma_db.add_documents(chunks, embedding_function=embeddings)
            st.success(f"Added documents from `{uploaded_file.name}` to the database.")
        except Exception as e:
            st.error(f"Error processing `{uploaded_file.name}`: {str(e)}")

# Function to initialize the Ollama LLM
@st.cache_resource
def initialize_llm():
    return Ollama(model=OLLAMA_MODEL)

# # Function to create the Retrieval QA chain
# @st.cache_resource
# def initialize_qa_chain(_chroma_db, _llm):
#     retriever = _chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#     return RetrievalQA.from_chain_type(llm=_llm, retriever=retriever, return_source_documents=True)

# Main App
def main():
    st.title("RAG Application with PDF Uploads")
    st.subheader("Powered by Streamlit, LangChain, Ollama, and Chroma")

    # Initialize Chroma database
    chroma_db = initialize_chroma()
    llm = initialize_llm()
    # qa_chain = initialize_qa_chain(chroma_db, llm)

    # Section for uploading PDFs
    st.sidebar.header("Upload PDF Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.sidebar.button("Process PDF Files"):
        if uploaded_files:
            with st.spinner("Processing uploaded PDFs..."):
                process_and_add_documents(uploaded_files, chroma_db)
        else:
            st.sidebar.warning("Please upload at least one PDF file.")

    # Query section
    st.header("Ask a Question")
    query = st.text_input("Enter your query:", placeholder="Ask me anything...")
    
    if st.button("Submit") and query.strip():
        with st.spinner("Retrieving relevant documents and generating a response..."):
            # Run the RAG pipeline
            # result = qa_chain(query)
            retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            relevant_docs = retriever.get_relevant_documents(query)
            
            prompt = PromptTemplate.from_template(
                """
                You are a helper for qeustion answering tasks. Use the following to answer the question.
                If you don't know the answer, just say you don't know. Use three sentences maximum and be concise in your answer.
                Question: {question}
                Context: {context}
                Answer:
                """
                )
                
            chain = ({"context": retriever, "question": RunnablePassthrough()}
                     | prompt
                     | llm
                     | StrOutputParser())
            
            result = chain.invoke(query)
            
            # response = result.get("result", "No response generated.")
            # source_documents = result.get("source_documents", [])
        
        # Display the response
        st.subheader("Response:")
        st.write(result)
        
        # Display the retrieved documents
        if relevant_docs:
            st.subheader("Retrieved Documents:")
            for i, doc in enumerate(relevant_docs):
                st.write(f"**Document {i + 1}:**")
                st.write(doc.page_content)
        else:
            st.write("No documents retrieved.")

# Run the app
if __name__ == "__main__":
    main()