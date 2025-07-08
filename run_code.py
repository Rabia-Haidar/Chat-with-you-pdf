import os
import shutil
import argparse
from dotenv import load_dotenv 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from query_data import query_rag

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--ask", type=str, help="Ask a question to your PDF.")
    args = parser.parse_args()
    if args.reset:
        print("\nClearing the database...\n")
        clear_database()

        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)

    if args.ask:
       query_rag(args.ask)
    else:
        print("Ready. Use `--ask \"Your question here\"` to query or `--reset` to rebuild.")

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    chunks_with_ids = calculate_chunk_ids(chunks)

    items = db.get(include=[])
    ids = set(items["ids"])
    print(f"Number of existing documents in DB: {len(ids)}")

    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks.")
        ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=ids)
        db.persist()
    else:
        print("No new documents to add.")

def calculate_chunk_ids(chunks):
    last_page_id = None
    chunk_id = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        page_id = f"{source}:{page}"
        
        if page_id == last_page_id:
            chunk_id += 1
        else:
            chunk_id = 0

        chunk.metadata["id"] = f"{page_id}:{chunk_id}"
        last_page_id = page_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Database at '{CHROMA_PATH}' cleared.")
    else:
        print(f"Database at '{CHROMA_PATH}' does not exist. Nothing to clear.")

if __name__ == "__main__":
    main()
