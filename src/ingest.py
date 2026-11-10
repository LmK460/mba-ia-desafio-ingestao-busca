"""Ingestão do PDF no PostgreSQL + pgVector.

Lê o PDF, divide em chunks de 1000 caracteres com overlap de 150,
gera os embeddings e salva tudo no banco vetorial.
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
    )


def ingest_pdf() -> None:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    print(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"{len(documents)} página(s) carregada(s).")

    print(f"Dividindo em chunks de {CHUNK_SIZE} caracteres com overlap de {CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} chunk(s) gerado(s).")

    print("Gerando embeddings e salvando no PostgreSQL/pgVector...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    print(f"Ingestão concluída! {len(chunks)} chunk(s) salvos na coleção '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()
