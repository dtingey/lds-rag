# lds-rag
Semantic Search and RAG-ish application for LDS Scriptures

Stack:
LitServe (by Lightning.AI)
Qdrant
Llama-3.1 (through Ollama)

Recommendation: Use uv!

## Setup

Create venv:
```bash
uv venv
```

Install prereqs
```bash
uv pip install -r requirements.txt
```

Create docker images:
```bash
docker compose -f vectordb_llm_server/docker-compose.yml up -d
```

Set url:
```bash
export LDS_RAG_URL="http://127.0.0.1:8000"
```

## Run it!

Run the server
```bash
python src/server.py
```

With the server running, run the client:
```bash
python src/scripture_client.py
```
