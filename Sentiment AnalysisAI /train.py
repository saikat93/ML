"""
Sentiment Analysis AI — 100% Local
====================================
- Embeddings  : sentence-transformers (all-MiniLM-L6-v2)
- Vector DB   : ChromaDB  (persistent, local, no cloud)
- Inference   : KNN cosine similarity + weighted majority vote
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = "./chroma_db"          # folder where ChromaDB persists data
COLLECTION     = "sentiment"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB, runs on CPU
TOP_K          = 5                      # neighbours for majority-vote

# ── Training data ─────────────────────────────────────────────────────────────
TRAINING_DATA = [
    # Positive
    ("I absolutely love this product! It exceeded all my expectations.", "positive"),
    ("What a wonderful experience. Truly delightful!", "positive"),
    ("The service was amazing and the staff were incredibly helpful.", "positive"),
    ("This is the best purchase I have ever made. Highly recommended!", "positive"),
    ("Fantastic quality and super fast delivery. Will buy again.", "positive"),
    ("I'm so happy with the results. Outstanding work!", "positive"),
    ("Great value for money. Really satisfied with everything.", "positive"),
    ("Brilliant product. Works perfectly and looks great too.", "positive"),
    ("Exceeded my expectations in every way. Five stars!", "positive"),
    ("Incredible experience. I can't stop recommending this to everyone.", "positive"),

    # Negative
    ("Terrible experience. I will never buy from here again.", "negative"),
    ("The product broke after one day. Complete waste of money.", "negative"),
    ("Horrible customer service. They ignored all my complaints.", "negative"),
    ("Very disappointing. Nothing like the description at all.", "negative"),
    ("Absolute garbage. Do not waste your time or money on this.", "negative"),
    ("I am extremely frustrated. This is the worst product ever.", "negative"),
    ("Poor quality and arrived damaged. Very unhappy.", "negative"),
    ("Awful experience from start to finish. Zero stars.", "negative"),
    ("Completely useless. Broke on first use and no refund given.", "negative"),
    ("Disgusting quality and rude staff. Will be complaining formally.", "negative"),

    # Neutral
    ("The package arrived on time and was as described.", "neutral"),
    ("It does what it says on the box. Nothing more, nothing less.", "neutral"),
    ("Average product. Neither good nor bad.", "neutral"),
    ("Delivery was okay. Product is acceptable for the price.", "neutral"),
    ("It meets basic requirements but nothing special.", "neutral"),
    ("Standard quality. Works as expected.", "neutral"),
    ("Nothing extraordinary but gets the job done.", "neutral"),
    ("Reasonable price for what you get. Not impressed but not upset.", "neutral"),
]


# ── ChromaDB setup ────────────────────────────────────────────────────────────
def get_collection():
    """Return a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


# ── Training ──────────────────────────────────────────────────────────────────
def train(data: list[tuple[str, str]] = TRAINING_DATA, reset: bool = False):
    """
    Embed training sentences and store in ChromaDB.

    Args:
        data:  list of (text, label) pairs
        reset: if True, wipe the collection before inserting
    """
    print(f"\n[Model] Loading '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    collection = get_collection()

    if reset:
        print("[ChromaDB] Resetting collection ...")
        client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(COLLECTION)
        collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    texts  = [t for t, _ in data]
    labels = [l for _, l in data]

    print(f"[Embed] Encoding {len(texts)} samples ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    ids = [f"train_{i}" for i in range(len(texts))]

    print("[ChromaDB] Upserting vectors ...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"sentiment": lbl} for lbl in labels],
    )

    count = collection.count()
    print(f"[Done] {count} vectors stored in ChromaDB at '{os.path.abspath(DB_PATH)}'.\n")
    return model, collection


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text: str, model: SentenceTransformer, collection, top_k: int = TOP_K):
    """
    Predict sentiment using KNN weighted majority vote.

    Returns:
        dict: sentiment, confidence, neighbours
    """
    embedding = model.encode([text], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    neighbours = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity: score = 1 - distance
        score = round(1 - dist, 4)
        neighbours.append({
            "text":      doc,
            "sentiment": meta["sentiment"],
            "score":     score,
        })

    # Weighted majority vote
    votes: dict[str, float] = {}
    for n in neighbours:
        votes[n["sentiment"]] = votes.get(n["sentiment"], 0) + n["score"]

    predicted  = max(votes, key=votes.get)
    total      = sum(votes.values())
    confidence = round(votes[predicted] / total * 100, 1) if total > 0 else 0.0

    return {
        "sentiment":  predicted,
        "confidence": confidence,
        "votes":      {k: round(v / total * 100, 1) for k, v in votes.items()},
        "neighbours": neighbours,
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, collection = train(reset=True)

    test_sentences = [
        "This movie was absolutely brilliant, I loved every second!",
        "Worst experience of my life. Never again.",
        "It arrived on time and works fine.",
        "The food was okay, not great but not terrible either.",
        "I am so disappointed. This product is a scam.",
    ]

    EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
    print("=" * 65)
    print("  SENTIMENT PREDICTIONS")
    print("=" * 65)

    for sentence in test_sentences:
        result = predict(sentence, model, collection)
        emoji  = EMOJI.get(result["sentiment"], "")
        print(f"\n📝 Text      : {sentence}")
        print(f"   Sentiment : {result['sentiment'].upper()} {emoji}  ({result['confidence']}% confidence)")
        print(f"   Votes     : {result['votes']}")
        print("   Top neighbours:")
        for n in result["neighbours"][:3]:
            print(f"     [{n['score']:.3f}] ({n['sentiment']:8s}) {n['text'][:65]}")

    print("\n" + "=" * 65)
