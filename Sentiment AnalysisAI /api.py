"""
FastAPI REST API — Local Sentiment Analysis
===========================================
No cloud. No API keys. ChromaDB + sentence-transformers only.

Endpoints:
  GET  /health      — service status
  POST /train       — (re)train and persist vectors to ChromaDB
  POST /predict     — predict sentiment for a given text
  GET  /stats       — collection stats
  POST /add         — add a single labelled example to the DB
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from train import train, predict, get_collection, EMBEDDING_MODEL, TRAINING_DATA

# ── Shared app state ──────────────────────────────────────────────────────────
state: dict = {"model": None, "collection": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + collection on startup."""
    print("[Startup] Loading model and ChromaDB collection ...")
    state["model"]      = SentenceTransformer(EMBEDDING_MODEL)
    state["collection"] = get_collection()
    print(f"[Startup] Ready. {state['collection'].count()} vectors in DB.")
    yield
    print("[Shutdown] Bye!")


app = FastAPI(
    title="Sentiment Analysis API (Local)",
    description="100% local sentiment analysis — sentence-transformers + ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="This product is amazing!")
    top_k: int = Field(5, ge=1, le=20)


class PredictResponse(BaseModel):
    sentiment:  str
    confidence: float
    votes:      dict[str, float]
    neighbours: list[dict]


class TrainRequest(BaseModel):
    reset: bool = Field(True, description="Wipe existing vectors before training")


class TrainResponse(BaseModel):
    message:        str
    vectors_stored: int


class AddRequest(BaseModel):
    text:      str = Field(..., min_length=1)
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    count = state["collection"].count() if state["collection"] else 0
    return {
        "status":        "ok",
        "model":         EMBEDDING_MODEL,
        "vectors_in_db": count,
    }


@app.get("/stats")
def stats():
    if not state["collection"]:
        raise HTTPException(503, "Collection not ready")
    col = state["collection"]
    all_items = col.get(include=["metadatas"])
    label_counts: dict[str, int] = {}
    for meta in all_items["metadatas"]:
        lbl = meta.get("sentiment", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    return {"total_vectors": col.count(), "by_sentiment": label_counts}


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    """Re-embed the built-in training set and persist to ChromaDB."""
    state["model"], state["collection"] = train(TRAINING_DATA, reset=req.reset)
    return {
        "message":        "Training complete.",
        "vectors_stored": state["collection"].count(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    """Predict sentiment of the given text."""
    if state["model"] is None or state["collection"] is None:
        raise HTTPException(503, "Model not ready. Call /train first.")
    if state["collection"].count() == 0:
        raise HTTPException(400, "No vectors in DB. Call /train first.")

    result = predict(req.text, state["model"], state["collection"], top_k=req.top_k)
    return result


@app.post("/add")
def add_example(req: AddRequest):
    """Add a single labelled example to the vector DB (online learning)."""
    if state["model"] is None or state["collection"] is None:
        raise HTTPException(503, "Model not ready.")

    import uuid
    embedding = state["model"].encode(
        [req.text], normalize_embeddings=True
    )[0].tolist()

    new_id = f"custom_{uuid.uuid4().hex[:8]}"
    state["collection"].upsert(
        ids=[new_id],
        embeddings=[embedding],
        documents=[req.text],
        metadatas=[{"sentiment": req.sentiment}],
    )
    return {
        "message": "Example added.",
        "id":      new_id,
        "total":   state["collection"].count(),
    }


# ── Run ───────────────────────────────────────────────────────────────────────
# uvicorn api:app --reload --port 8000
