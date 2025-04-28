import os
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

data_folder = Path('./data')
docs = []

# Iterar solo sobre archivos PDF en la carpeta
for pdf_file in data_folder.glob('*.pdf'):
    docs.extend(PyPDFLoader(str(pdf_file)).load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

# Create vectorstore with OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="pdf-chroma-store",
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma"
)