# Ingestão e Busca Semântica com LangChain e PostgreSQL + pgVector

Desafio do módulo **Introdução ao LangChain** (MBA Full Cycle).

Sistema RAG em dois fluxos:

1. **Ingestão** (`src/ingest.py`): lê o `document.pdf`, divide em chunks de **1000 caracteres** com **overlap de 150**, gera os embeddings (Gemini) e grava os vetores no **PostgreSQL com pgVector**.
2. **Busca / Chat** (`src/chat.py` + `src/search.py`): CLI onde o usuário faz perguntas. A pergunta é vetorizada, os **10 trechos mais relevantes** são buscados no banco vetorial (`similarity_search_with_score(query, k=10)`), o prompt com as regras é montado e enviado à LLM, que responde **apenas com base no conteúdo do PDF**.

## Tecnologias

- **Python 3.10+**
- **LangChain** (`langchain`, `langchain-community`, `langchain-google-genai`, `langchain-postgres`, `langchain-text-splitters`)
- **PostgreSQL 17 + pgVector** via **Docker / Docker Compose**
- **Google Gemini** para embeddings e LLM

## Modelos utilizados

| Papel | Modelo | Observação |
| --- | --- | --- |
| Embeddings | `models/gemini-embedding-001` | Text-only, 3072 dimensões |
| LLM | `gemini-3.5-flash-lite` | Modelo leve/mais barato |

Ambos são configuráveis pelas variáveis de ambiente `GOOGLE_EMBEDDING_MODEL` e `GOOGLE_LLM_MODEL` no `.env`.

> **Atenção:** modelos de embeddings diferentes geram vetores com dimensões diferentes. A tabela de vetores é criada na primeira ingestão já com a dimensão do modelo escolhido. Se trocar de modelo de embeddings, apague a coleção existente (ou o volume do banco) e refaça a ingestão do zero — veja [Resetar o banco](#resetar-o-banco).

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API Key do Google (Gemini): crie gratuitamente em <https://aistudio.google.com/apikey>

## Estrutura do projeto

```
├── docker-compose.yml    # PostgreSQL + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Busca vetorial + prompt + LLM
│   └── chat.py           # CLI para interação com o usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Este arquivo
```

## Como executar

### 1. Criar o ambiente virtual e instalar as dependências

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configurar as variáveis de ambiente

Copie o template e informe sua API key do Google AI Studio:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
GOOGLE_API_KEY=sua-api-key-aqui
```

### 3. Subir o banco de dados

```bash
docker compose up -d
```

Isso sobe um PostgreSQL 17 com a extensão `pgVector` habilitada, exposto em `localhost:5432` (usuário `postgres`, senha `postgres`, banco `rag`).

### 4. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Carregando PDF: document.pdf
N página(s) carregada(s).
Dividindo em chunks de 1000 caracteres com overlap de 150...
M chunk(s) gerado(s).
Gerando embeddings e salvando no PostgreSQL/pgVector...
Ingestão concluída! M chunk(s) salvos na coleção 'pdf_documents'.
```

### 5. Rodar o chat

```bash
python src/chat.py
```

## Exemplo de uso

Pergunta dentro do contexto:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Pergunta fora do contexto:

```
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` para encerrar o chat.

## Resetar o banco

Caso precise refazer a ingestão do zero (ex.: troca de modelo de embeddings, ingestão duplicada):

```bash
docker compose down -v   # remove o container e o volume com os vetores
docker compose up -d     # sobe o banco limpo
python src/ingest.py     # refaz a ingestão
```
