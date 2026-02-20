# L88 Local Agentic RAG

Production-oriented local Agentic RAG system with:
- Hybrid retrieval (BGE embeddings + FAISS + BM25 + RRF + reranker)
- Strong agent graph (PLAN -> ACT -> VERIFY -> SYNTHESIZE, max 2 retries)
- Local vLLM inference (FP16)
- FastAPI backend (`/query`, `/upload`, `/reindex`, `/health`)

## Quick start

```bash
cd L88
pip install -r requirements.txt
export PYTHONPATH=$PWD
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Ingestion

```bash
python scripts/ingest.py ./data/my_docs
```

## Test

```bash
pytest -q
```
