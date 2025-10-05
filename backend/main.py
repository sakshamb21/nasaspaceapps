from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from summarizer_service import summarize
from semantic_retrieval import NASAStudySemanticRetriever

app = FastAPI(title="NASA Bioscience Dashboard API")

# ---- CORS (open for dev, restrict for prod) ----
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

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/search")
def search(query: str, top_k: int = 5):
    """Return top-k relevant studies for a query."""
    try:
        df = retriever.search(query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return df.to_dict(orient="records")

# Summarizer endpoint already in summarizer_service.py
# We just re-export it here so the frontend hits /api/summarize
from summarizer_service import summarize as summarize_endpoint
app.post("/api/summarize")(summarize_endpoint)
