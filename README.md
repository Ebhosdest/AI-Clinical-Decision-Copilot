# AI Clinical Decision Copilot

Evidence-based clinical decision support powered by RAG + LangGraph multi-agent reasoning.

## Architecture

```
Clinical Query
      │
      ▼
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Streamlit  │────▶│              FastAPI (Port 8000)              │
│  Frontend   │◀────│  POST /query  POST /query/stream  GET /health │
└─────────────┘     └──────────────────────┬───────────────────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │         LangGraph Pipeline        │
                          │                                   │
                          │  ┌─────────┐   ┌─────────────┐   │
                          │  │ Triage  │──▶│  Retrieval  │   │
                          │  │  Agent  │   │    Agent    │   │
                          │  └────┬────┘   └──────┬──────┘   │
                          │       │ (fast-track)   │          │
                          │       └────────────────┘          │
                          │                │                  │
                          │         ┌──────▼──────┐           │
                          │         │  Reasoning  │           │
                          │         │    Agent    │           │
                          │         └──────┬──────┘           │
                          │                │                  │
                          │         ┌──────▼──────┐           │
                          │         │  Response   │           │
                          │         │    Agent    │           │
                          │         └─────────────┘           │
                          └──────────────┬───────────────────┘
                                         │
                          ┌──────────────▼───────────────────┐
                          │          RAG Pipeline             │
                          │  FAISS (cosine) + OpenAI embeds   │
                          │  AHA-ACS | WHO-HTN | ADA-DM docs  │
                          └──────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
echo "OPENAI_API_KEY=sk-..." >> .env

# 3. Build the vector index
python -m src.rag.pipeline build

# 4. Start the API
uvicorn src.api.main:app --reload

# 5. Launch the UI
streamlit run frontend/app.py
```

## Project Structure

```
.
├── src/
│   ├── config.py               # Centralised config (env-based)
│   ├── rag/
│   │   ├── loader.py           # Document loader + section-aware chunker
│   │   ├── embeddings.py       # OpenAI embedding service
│   │   ├── vector_store.py     # FAISS IndexFlatIP wrapper
│   │   ├── retriever.py        # Query → search → formatted context
│   │   ├── generator.py        # GPT-4o response generation + streaming
│   │   └── pipeline.py         # RAG orchestrator (build + query)
│   ├── agents/
│   │   ├── state.py            # AgentState TypedDict
│   │   ├── triage_agent.py     # Urgency classification + patient extraction
│   │   ├── retrieval_agent.py  # Multi-query evidence retrieval
│   │   ├── reasoning_agent.py  # Differential diagnosis reasoning
│   │   ├── response_agent.py   # Final response synthesis
│   │   └── graph.py            # LangGraph StateGraph workflow
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── models.py           # Pydantic schemas
│       └── dependencies.py     # Singleton injection
├── frontend/
│   └── app.py                  # Streamlit UI
├── data/
│   └── guidelines/             # Clinical guideline .txt/.pdf files
├── tests/
│   ├── test_rag.py             # RAG unit tests (no API key needed)
│   └── test_eval.py            # Retrieval evaluation suite
├── .env                        # API keys (not committed)
└── requirements.txt
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| POST   | `/query`          | Clinical query (agent or RAG mode)   |
| POST   | `/query/stream`   | Streaming SSE response               |
| POST   | `/index/build`    | Rebuild FAISS index from guidelines  |
| GET    | `/health`         | Service health + index status        |

## Tech Stack

- **LLM:** OpenAI GPT-4o
- **Embeddings:** text-embedding-3-small (1536d)
- **Vector Store:** FAISS IndexFlatIP (cosine similarity)
- **Agent Orchestration:** LangGraph StateGraph
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Config:** python-dotenv + Pydantic

## Running Tests

```bash
# All tests (no API key required for unit + structural tests)
python -m pytest tests/ -v

# Include live retrieval evaluation (requires OPENAI_API_KEY)
python -m pytest tests/test_eval.py -v -k "Live"
```

## Adding Guidelines

Drop `.txt` or `.pdf` files into `data/guidelines/`. Each file should include:

```
SOURCE: <Institution Name>
DOCUMENT_ID: <UNIQUE-ID-YEAR>

SECTION: <Section Name>
<Section content...>
```

Rebuild the index after adding files: `python -m src.rag.pipeline build`
