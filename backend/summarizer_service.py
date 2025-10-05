import os
import time
import logging
from typing import List, Dict, Any
from functools import lru_cache
import dotenv

# Import the Google GenAI SDK client
from google import genai
from google.genai import types
from google.genai.errors import APIError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# Ensure pandas is available for the retriever's DataFrame operations
import pandas as pd 

# Import your retriever (from semantic_retrieval.py you already built)
# NOTE: You must have 'semantic_retrieval.py' in your path.
# from semantic_retrieval import NASAStudySemanticRetriever 

# ---- Configuration ----
# SECURITY FIX: Do NOT hardcode the API key (removed hardcoded fallback key)

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CSV_PATH = os.getenv("CSV_PATH", "nasa_bioscience_processed1.csv")
EMBEDDING_FILE = os.getenv("EMBEDDING_FILE", "embeddings.npy")

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("summarizer")

# ---- Initialize Gemini Client (Centralized and Cached) ----
@lru_cache(maxsize=1)
def get_gemini_client():
    """Initializes and returns a configured Gemini client."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        raise ValueError("API Key is missing. Set the GEMINI_API_KEY environment variable.")
    
    try:
        # The client automatically uses the GEMINI_API_KEY environment variable
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"Gemini client initialized for model: {MODEL_NAME}")
        return client
    except Exception as e:
        logger.exception("Failed to initialize Gemini client.")
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# ---- FastAPI app ----
app = FastAPI(title="NASA Bioscience Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # tighten to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request/Response Models ----
class SummarizeRequest(BaseModel):
    query: str
    audience: str = "scientist"
    top_k: int = 5

class SummarizeResponse(BaseModel):
    query: str
    audience: str
    summary: str
    sources: List[str]

# ---- Audience instructions ----
SYSTEM_INSTRUCTIONS = {
    "scientist": (
        "You are a bioscience research assistant. Summarize for a scientist: include experimental setup, quantitative results, "
        "biological implications, and reliability. Use precise language and cite PCMIDs inline as [PCMID]."
    ),
    "manager": (
        "You are a succinct research briefing writer. Summarize for a program manager: focus on main findings, addressed gaps, "
        "funding/next steps, and practical implications. Avoid technical jargon; keep it actionable and clear."
    ),
    "architect": (
        "You are a mission safety/operations analyst. Summarize for a mission architect: highlight operationally relevant results, "
        "direct safety/engineering implications, suggested mitigations or tests, and confidence levels."
    ),
}

# ---- Retriever init (Placeholder/Safety Check) ----
@lru_cache(maxsize=1)
def get_retriever():
    logger.info("Initializing semantic retriever...")
    # NOTE: Temporarily using a placeholder if 'semantic_retrieval' is not yet defined.
    # Replace this with your actual working retriever initialization.
    try:
        from semantic_retrieval import NASAStudySemanticRetriever
        return NASAStudySemanticRetriever(csv_path=CSV_PATH, embedding_file=EMBEDDING_FILE)
    except ImportError:
        logger.warning("NASAStudySemanticRetriever not found. Using dummy retriever.")
        class DummyRetriever:
            def search(self, query, top_k):
                return pd.DataFrame([
                    {"PCMID": "30000001", "Title": "Mock Study on Space Radiation", "Abstract": "Placeholder abstract. Radiation exposure observed to increase gene expression in yeast.", "Link": "http://example.com/30000001"},
                ])
        return DummyRetriever()

# ---- Prompt builder (Standard RAG Context) ----
def build_prompt(audience: str, query: str, docs: List[Dict[str, Any]]) -> str:
    """Builds the prompt string containing the user query and retrieved context."""
    
    # We will pass the system instruction separately to the SDK,
    # so the prompt only contains the context and the user query instruction.
    
    parts = [
        f"User Query: {query}\n\n",
        "Retrieved Studies (use this context to formulate your response):\n"
    ]
    
    # Concatenate document context
    for d in docs:
        abstract = d.get('Abstract', '')
        # Truncate abstracts to prevent exceeding context window limits
        truncated_abstract = abstract[:1500] + "..." if len(abstract) > 1500 else abstract
        entry = (
            f"[PCMID: {d.get('PCMID','')}]\n"
            f"Title: {d.get('Title','')}\n"
            f"Abstract: {truncated_abstract}\n"
            f"Link: {d.get('Link','')}\n\n"
        )
        parts.append(entry)
        
    parts.append("End of Retrieved Studies. Now, generate the audience-specific summary based *only* on the provided context.")
    return "\n".join(parts)


# ---- Summarization endpoint (UPDATED) ----
@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    # 1. Initialization and Context Retrieval
    try:
        client = get_gemini_client()
        retriever = get_retriever()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) # Catches missing API key

    try:
        results_df = retriever.search(req.query, top_k=req.top_k)
    except Exception as e:
        logger.exception("Error during retrieval")
        raise HTTPException(status_code=500, detail="Error during document retrieval")

    docs = []
    for _, row in results_df.iterrows():
        docs.append({
            "PCMID": str(row.get("PCMID", "")),
            "Title": str(row.get("Title", "")),
            "Abstract": str(row.get("Abstract", "")),
            "Link": str(row.get("Link", "")),
        })

    prompt_content = build_prompt(req.audience, req.query, docs)
    system_instruction = SYSTEM_INSTRUCTIONS.get(req.audience.lower(), SYSTEM_INSTRUCTIONS["scientist"])

    # 2. Gemini API Call
    try:
        # Use the SDK with the correct System Instruction and Model configuration
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            # Set max output tokens if needed, e.g., max_output_tokens=512
            # Set temperature for creativity, e.g., temperature=0.0 to 1.0
        )
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_content], # The prompt contains the user query and context
            config=config
        )
        
        summary_text = response.text.strip()
        
    except APIError as e:
        logger.error(f"Gemini API Error: {e.status_code} - {e.message}")
        raise HTTPException(status_code=502, detail=f"Gemini API failed: {e.message}")
    except Exception as e:
        logger.exception("Gemini API call failed with unexpected error.")
        raise HTTPException(status_code=502, detail=f"LLM summarization failed: {e}")

    # 3. Response Generation
    if not summary_text:
        # This catches cases where the model response is empty, potentially due to safety filtering
        if response.candidates and response.candidates[0].finish_reason != types.FinishReason.STOP:
             raise HTTPException(status_code=502, detail=f"LLM response was incomplete or blocked by safety settings (Reason: {response.candidates[0].finish_reason.name}).")
        else:
             raise HTTPException(status_code=502, detail="LLM returned an empty summary.")


    return SummarizeResponse(
        query=req.query,
        audience=req.audience,
        summary=summary_text,
        sources=[d["PCMID"] for d in docs],
    )

# ---- Health check ----
@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}
