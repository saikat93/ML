# 🤖 Local RAG System
**Stack:** Ollama · ChromaDB · nomic-embed-text · llama3

Fully local — no cloud, no API keys, your data never leaves your machine.

---

## ⚙️ 1. Prerequisites

### Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

### Pull the required models
```bash
ollama pull llama3               # LLM for answering questions (~4.7 GB)
ollama pull nomic-embed-text     # Embedding model (~274 MB)
```

> **Alternative LLMs** (lighter): `ollama pull mistral` or `ollama pull phi3`
> Change `LLM_MODEL` in `rag.py` to match.

---

## 📦 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

Requires Python 3.10+

---

## 🚀 3. Usage

### Ingest a PDF
```bash
python rag.py ingest path/to/your_document.pdf
```
You can ingest multiple PDFs — they all go into the same knowledge base.

### Ask a single question
```bash
python rag.py query "What are the main conclusions?"
python rag.py query "Summarise the methodology section"
```

### Interactive chat
```bash
python rag.py chat
```
Type your questions and get grounded answers. Type `exit` to quit.

---

## ⚙️ Configuration (top of rag.py)

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHROMA_DIR` | `./chroma_db` | Where ChromaDB saves data |

---

## 🗂️ Project structure
```
rag_system/
├── rag.py            ← main script
├── requirements.txt
├── README.md
├── chroma_db/        ← created automatically on first ingest
└── docs/             ← put your PDFs here (optional)
```

---

## 🛠️ Troubleshooting

**`Connection refused` on embedding/generation**
→ Make sure Ollama is running: `ollama serve`

**`model not found`**
→ Pull the model first: `ollama pull llama3`

**Slow on first query after ingest**
→ Ollama loads the model into RAM on first use. Subsequent queries are fast.

**Scanned PDF returns no text**
→ Scanned PDFs are images; they need OCR first.
   Install `ocrmypdf` and run: `ocrmypdf input.pdf output.pdf` then ingest `output.pdf`.
