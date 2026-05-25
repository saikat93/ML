# 🧠 Sentiment Analysis AI — 100% Local

No cloud. No API keys. No internet after setup.

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | `ChromaDB` (persistent, local folder) |
| Inference | KNN cosine similarity + weighted majority vote |
| API | `FastAPI` + `uvicorn` |

---

## 📁 Project Structure

```
sentiment_ai_local/
├── train.py          # Core: embed → store in ChromaDB → predict
├── api.py            # FastAPI REST service
├── cli.py            # Interactive terminal predictor
├── requirements.txt
├── README.md
└── chroma_db/        # ← created automatically on first train
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Download `all-MiniLM-L6-v2` once (~90 MB)
- Create a `./chroma_db/` folder with persistent vectors
- Embed 28 labelled sentences and store them
- Run 5 test predictions

### 3. Interactive CLI
```bash
python cli.py
```
Type any text and get instant coloured predictions.

---

## 🌐 REST API

### Start
```bash
uvicorn api:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status + vector count |
| GET | `/stats` | Breakdown by sentiment |
| POST | `/train` | Re-train on built-in data |
| POST | `/predict` | Predict sentiment |
| POST | `/add` | Add a custom labelled example |

### Example — predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is absolutely amazing!", "top_k": 5}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 96.3,
  "votes": {"positive": 96.3, "neutral": 3.7},
  "neighbours": [
    {"text": "...", "sentiment": "positive", "score": 0.934},
    ...
  ]
}
```

### Example — add your own training example
```bash
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"text": "The colour is off but it works fine.", "sentiment": "neutral"}'
```

---

## 🔧 How It Works

```
Training:
  Text  ──► SentenceTransformer ──► 384-dim vector ──► ChromaDB upsert
  Label ──────────────────────────────────────────────► stored as metadata

Inference:
  Query ──► SentenceTransformer ──► 384-dim vector
         ──► ChromaDB cosine KNN search
         ──► Top-K neighbours + similarity scores
         ──► Weighted majority vote
         ──► Sentiment label + confidence %
```

---

## ➕ Add Your Own Data

Edit `TRAINING_DATA` in `train.py`:
```python
TRAINING_DATA = [
    ("Your sentence here", "positive"),
    ("Another sentence",   "negative"),
    ("Meh, it was okay",   "neutral"),
    ...
]
```
Then re-run `python train.py`.

Or add examples live without retraining:
```bash
python -c "
from train import get_collection, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(EMBEDDING_MODEL)
emb = m.encode(['Great build quality!'], normalize_embeddings=True)[0].tolist()
get_collection().upsert(ids=['my_1'], embeddings=[emb], documents=['Great build quality!'], metadatas=[{'sentiment':'positive'}])
print('Added!')
"
```

---

## 💡 Tips

- **More accuracy** → add more training examples (50+ per class is ideal)
- **Different domain** → replace training data with domain-specific sentences
- **Bigger model** → change `EMBEDDING_MODEL` to `all-mpnet-base-v2` and set `EMBED_DIM=768`
- **ChromaDB data** lives in `./chroma_db/` — back it up or delete to reset
