# backend/main_graph.py
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from summarizer_service import summarize as summarize_endpoint
from semantic_retrieval import NASAStudySemanticRetriever
from graph_integration import ingest_summarizer_output, fetch_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_graph")

app = FastAPI(title="NASA Bioscience Dashboard API")

# ---- CORS (open for dev) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Global retriever instance ----
retriever = NASAStudySemanticRetriever(
    csv_path="nasa_bioscience_processed.csv",
    embedding_file="embeddings.npy"
)

# -------------------------------
# Health check
# -------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}

# -------------------------------
# Search endpoint
# -------------------------------
@app.get("/api/search")
def search(query: str, top_k: int = 5):
    """Return top-k relevant studies for a query."""
    try:
        df = retriever.search(query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return df.to_dict(orient="records")

# -------------------------------
# Summarizer endpoint
# -------------------------------
app.post("/api/summarize")(summarize_endpoint)

# -------------------------------
# Graph update endpoint
# -------------------------------
@app.post("/api/graph/update")
def update_graph(payload: dict):
    """
    Payload example:
    {
        "id": "PMC123456",
        "title": "Sample study",
        "abstract": "This study investigates...",
    }
    """
    try:
        if not payload.get("id") or not payload.get("abstract"):
            raise HTTPException(status_code=400, detail="Missing study id or abstract")

        result = ingest_summarizer_output(payload)
        return {"status": "ok", "detail": result}

    except Exception as e:
        logger.exception("Graph update failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Graph fetch endpoint
# -------------------------------
@app.get("/api/graph")
def get_graph(node_id: str, node_type: Optional[str] = "Study", radius: int = 2):
    """
    Fetch graph JSON (nodes + links) for a given node.
    node_type: "Study" or "Query"
    """
    try:
        graph_data = fetch_graph(node_id=node_id, node_type=node_type, radius=radius)
        if not graph_data["nodes"]:
            raise HTTPException(status_code=404, detail="No graph data found for given node")
        return graph_data
    except Exception as e:
        logger.exception("Graph fetch failed")
        raise HTTPException(status_code=500, detail=str(e))
