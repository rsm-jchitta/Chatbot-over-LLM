import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Setup Streamlit title and sidebar
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input for URLs in the sidebar
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

# Button to process URLs in the sidebar
process_url_clicked = st.sidebar.button("Process URLs")

# Placeholder for the main content
main_placeholder = st.container()

# Define your OpenAI instance with API Key
llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key='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')

if process_url_clicked:
    # Filter out empty URLs
    urls = [url for url in urls if url.strip()]
    if urls:
        # Load and process data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.write("Data Loading...Started...✅✅✅")
        data = loader.load()
        if data:
            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            if docs:
                # Create embeddings and save them to a FAISS index
                embeddings = OpenAIEmbeddings(openai_api_key='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')
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                time.sleep(2)

                # Save the FAISS index to a pickle file
                file_path = "your_file_path_here"  # Specify your file path
                vectorstore_openai.save_local(file_path)
                main_placeholder.text(f"FAISS index saved to {file_path}. ✅")
            else:
                main_placeholder.text("Text Splitter produced empty documents. Check data.")
        else:
            main_placeholder.text("Data loading failed. Check URLs or network connection.")
    else:
        main_placeholder.text("Please enter at least one URL.")

# Question input at the center of the page
with main_placeholder:
    query = st.text_input("Question: ")
    if query and 'retriever' in st.session_state:
        r = st.session_state['retriever']
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=r)
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])
        
        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.write("Please process URLs first to load and index the data.")
