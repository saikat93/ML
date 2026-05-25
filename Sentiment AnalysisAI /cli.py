"""
Interactive CLI — type text and get instant sentiment predictions.
Run after training: python cli.py
"""

from sentence_transformers import SentenceTransformer
from train import predict, get_collection, EMBEDDING_MODEL

EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
COLORS = {
    "positive": "\033[92m",  # green
    "negative": "\033[91m",  # red
    "neutral":  "\033[93m",  # yellow
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "dim":      "\033[2m",
}

def colored(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def main():
    print(colored("\n🧠 Sentiment Analysis CLI — 100% Local", "bold"))
    print(colored("   sentence-transformers + ChromaDB", "dim"))
    print(colored("   Type 'quit' or 'exit' to stop.\n", "dim"))

    print("[Model] Loading embedding model ...")
    model      = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_collection()

    if collection.count() == 0:
        print(colored("\n⚠️  No vectors found in ChromaDB.", "negative"))
        print("   Run  python train.py  first to train the model.\n")
        return

    print(colored(f"[Ready] {collection.count()} vectors loaded from ChromaDB.\n", "positive"))

    while True:
        try:
            text = input(colored("Enter text > ", "bold")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result    = predict(text, model, collection)
        sentiment = result["sentiment"]
        emoji     = EMOJI.get(sentiment, "")
        color     = sentiment  # maps directly to color keys

        print(f"\n  Result    : {colored(sentiment.upper(), color)} {emoji}")
        print(f"  Confidence: {colored(str(result['confidence']) + '%', color)}")
        print(f"  Votes     : ", end="")
        for lbl, pct in result["votes"].items():
            print(f"{lbl}={pct}%  ", end="")
        print()

        print(colored("  Top matches:", "dim"))
        for n in result["neighbours"][:3]:
            c = n["sentiment"]
            print(colored(f"    [{n['score']:.3f}] ({n['sentiment']:8s}) {n['text'][:60]}", "dim"))
        print()


if __name__ == "__main__":
    main()
