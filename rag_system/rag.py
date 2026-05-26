"""
Local RAG System
================
Stack: Ollama (LLM) + ChromaDB (vector store) + nomic-embed-text (embeddings)
Usage:
    python rag.py ingest path/to/your.pdf
    python rag.py query "What is the document about?"
    python rag.py chat   (interactive mode)
"""

import sys
import os
import argparse
import textwrap
from pathlib import Path

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    import ollama
    import pypdf
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.progress import track
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run:  pip install chromadb ollama pypdf rich")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"          # where ChromaDB stores data on disk
COLLECTION     = "rag_docs"             # collection name inside ChromaDB
EMBED_MODEL    = "nomic-embed-text"     # Ollama embedding model
LLM_MODEL      = "phi3:mini"           # Ollama chat model (change to mistral, phi3, etc.)
CHUNK_SIZE     = 500                    # characters per chunk
CHUNK_OVERLAP  = 80                     # overlap between consecutive chunks
TOP_K          = 4                      # how many chunks to retrieve per query

console = Console()

# ── ChromaDB setup ────────────────────────────────────────────────────────────
def get_collection():
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Use Ollama embeddings via ChromaDB's embedding function wrapper
    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

# ── PDF loading ───────────────────────────────────────────────────────────────
def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    text = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 30]   # drop tiny tail chunks

# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest(pdf_path: str):
    """Load a PDF, chunk it, embed chunks, store in ChromaDB."""
    path = Path(pdf_path)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {pdf_path}")
        sys.exit(1)

    console.print(f"\n[bold cyan]Ingesting:[/bold cyan] {path.name}")

    # 1. Extract text
    console.print("  • Extracting text from PDF…")
    text = load_pdf(str(path))
    if not text.strip():
        console.print("[red]No text found in PDF. It may be a scanned image-only PDF.[/red]")
        sys.exit(1)
    console.print(f"    Extracted {len(text):,} characters")

    # 2. Chunk
    chunks = chunk_text(text)
    console.print(f"  • Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 3. Store in ChromaDB (embeddings generated automatically via Ollama)
    collection = get_collection()
    doc_name = path.stem

    # Build unique IDs so re-ingesting the same file replaces old chunks
    ids       = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]

    console.print("  • Embedding & storing chunks (this may take a minute)…")
    # Upsert in batches of 50 to avoid memory spikes
    batch = 50
    for i in track(range(0, len(chunks), batch), description="  Embedding"):
        collection.upsert(
            ids=ids[i:i+batch],
            documents=chunks[i:i+batch],
            metadatas=metadatas[i:i+batch],
        )

    console.print(f"\n[bold green]✓ Done![/bold green] {len(chunks)} chunks from '{path.name}' stored in ChromaDB.\n")

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k most relevant chunks for a query."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "source": meta["source"], "score": 1 - dist})
    return hits

# ── Generation ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Answer only from the context below. Do NOT use outside knowledge.
- If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to answer that."
- Be concise but complete.
- Cite the source file name when relevant.
"""

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"[{i}] (from {c['source']}, relevance {c['score']:.2f}):\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {query}"

def ask(query: str, show_sources: bool = True) -> str:
    """Retrieve context and generate an answer via Ollama."""
    # 1. Retrieve
    chunks = retrieve(query)
    if not chunks:
        return "No documents found in the knowledge base. Please ingest a PDF first."

    # 2. Build prompt
    user_prompt = build_prompt(query, chunks)

    # 3. Generate
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    answer = response["message"]["content"]

    # 4. Optionally show sources
    if show_sources:
        sources = "\n".join(
            f"  [{i+1}] {c['source']} (score: {c['score']:.2f})"
            for i, c in enumerate(chunks)
        )
        answer += f"\n\n---\n**Sources:**\n{sources}"

    return answer

# ── Interactive chat ──────────────────────────────────────────────────────────
def chat_loop():
    console.print(Panel(
        "[bold]Local RAG Chat[/bold]\n"
        f"Model: [cyan]{LLM_MODEL}[/cyan]  |  "
        f"Embeddings: [cyan]{EMBED_MODEL}[/cyan]  |  "
        f"Vector DB: [cyan]ChromaDB[/cyan]\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
        title="🤖 RAG System",
        border_style="cyan",
    ))

    while True:
        try:
            query = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if not query:
            continue

        with console.status("[cyan]Thinking…[/cyan]"):
            answer = ask(query)

        console.print("\n[bold green]Assistant[/bold green]")
        console.print(Markdown(answer))

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Local RAG system — Ollama + ChromaDB + PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python rag.py ingest report.pdf
              python rag.py query "What are the key findings?"
              python rag.py chat
        """),
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into the knowledge base")
    p_ingest.add_argument("pdf", help="Path to the PDF file")

    # query
    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", help="Your question in quotes")
    p_query.add_argument("--no-sources", action="store_true", help="Hide source references")

    # chat
    sub.add_parser("chat", help="Start an interactive chat session")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf)

    elif args.command == "query":
        with console.status("[cyan]Thinking…[/cyan]"):
            answer = ask(args.question, show_sources=not args.no_sources)
        console.print(Markdown(answer))

    elif args.command == "chat":
        chat_loop()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
