"""Busca semântica no PostgreSQL + pgVector e montagem do prompt para a LLM.

Fluxo: vetoriza a pergunta -> busca os 10 trechos mais relevantes no banco
vetorial -> monta o prompt com o contexto e as regras -> chama a LLM.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite")

TOP_K = 10

DEFAULT_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
    )


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=LLM_MODEL)


def search_context(question: str, k: int = TOP_K) -> str:
    """Vetoriza a pergunta, busca os k trechos mais relevantes e retorna o contexto concatenado."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=k)

    if not results:
        return ""

    return "\n\n---\n\n".join(doc.page_content for doc, _score in results)


def search_prompt(question: str) -> str | None:
    """Monta o prompt final com o contexto recuperado do banco vetorial.

    Retorna None quando não há contexto (banco vazio / ingestão não executada).
    """
    contexto = search_context(question)

    if not contexto:
        return None

    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)


def answer_question(question: str) -> str:
    """Executa o fluxo de RAG: busca -> prompt -> LLM -> resposta."""
    prompt = search_prompt(question)

    if prompt is None:
        print("Aviso: nenhum resultado encontrado no banco vetorial. Execute `python src/ingest.py` primeiro.")
        return DEFAULT_ANSWER

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()
