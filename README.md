# lowram-ollama-raft-rag

A RAG pipeline and RAFT-style training dataset generator built for running on **low-RAM, CPU-only hardware** — no cloud models, no big GPU required. Built around a self-hosted [Ollama](https://ollama.com) instance and a locally quantized model.

## What this does

This project gives a local Ollama model the ability to answer questions grounded in real documentation instead of guessing from memory. It works in two stages:

1. **RAG (Retrieval-Augmented Generation)** — documentation gets chunked, embedded, and stored in a local ChromaDB vector database. At query time, relevant chunks are retrieved and handed to the model as context.
2. **RAFT dataset generation** — a step beyond plain RAG. For a set of questions, the pipeline retrieves a relevant chunk *and* a distractor chunk from an unrelated source, then generates an answer that must cite the correct source and ignore the distractor. The output is a small `.jsonl` dataset intended as a first step toward future fine-tuning (not included in this repo — this repo stops at dataset generation).

Everything runs locally. Nothing is sent to any cloud API.

## Why it matters

Small local coding/chat models tend to guess at tool-specific details (CLI flags, API parameters, config syntax) rather than admit uncertainty. Grounding answers in real docs — and building a dataset that explicitly rewards ignoring irrelevant context — is a low-cost way to make a small local model noticeably more reliable, without needing enterprise hardware.

## What you need

- Linux (developed and tested on Fedora; should work on any distro with Python 3.10+)
- [Ollama](https://ollama.com) installed and running
- Python 3.10+
- At least 8GB RAM (developed on a 13GB RAM, CPU-only machine — no dedicated GPU required)
- Any Ollama-compatible chat model, plus an embedding model (this project uses `nomic-embed-text`)

## Setup

```bash
git clone https://github.com/tomey265/lowram-ollama-raft-rag.git
cd lowram-ollama-raft-rag

python3 -m venv venv
source venv/bin/activate

pip install chromadb ollama
```

Pull the embedding model (small, ~274MB):

```bash
ollama pull nomic-embed-text
```

Pull or point to your own chat model. This project was built and tested against a custom quantized Qwen2 coding model; substitute your own model name in the config section of `query.py` and `make_raft_dataset.py`.

## Usage

Run each step in order:

```bash
# 1. Download the documentation set into ./docs
python fetch_docs.py

# 2. Chunk, embed, and store the docs in ChromaDB
python ingest.py

# 3. Ask a question against the RAG pipeline
python query.py "How do I reduce Ollama's memory usage?"

# 4. Generate a small RAFT-style training dataset
python make_raft_dataset.py
```

### Adding your own documents

Drop additional `.md` or `.txt` files into `./docs`, then re-run `ingest.py`. To point this project at an entirely different documentation set, edit the `DOCUMENTS` dictionary in `fetch_docs.py`.

### Changing the RAFT questions

Edit the `QUESTIONS` list at the top of `make_raft_dataset.py`.

## What the output looks like

Each line of `raft_dataset.jsonl` is one JSON record:

```json
{
  "question": "How do I inspect the Modelfile of my-model:latest?",
  "relevant_context": {
    "source": "modelfile_reference.md",
    "chunk_index": 2,
    "text": "..."
  },
  "distractor_context": [
    {
      "source": "api_streaming.md",
      "chunk_index": 0,
      "text": "..."
    }
  ],
  "answer": "To inspect a model's Modelfile: ollama show --modelfile my-model:latest\n\n(modelfile_reference.md)",
  "generator_model": "your-model:latest",
  "embedding_model": "nomic-embed-text:latest",
  "created_at_utc": "2026-08-13T12:24:00+00:00"
}
```

## Low-RAM tuning notes

A few environment variables matter a lot on constrained hardware — this project uses `keep_alive="0s"` throughout so models unload from memory immediately after each call. Worth knowing about if you're tuning this for your own machine:

- `OLLAMA_FLASH_ATTENTION=1` — reduces memory use as context grows
- `OLLAMA_KV_CACHE_TYPE=q8_0` — meaningfully cuts KV cache memory with minimal quality loss (`q4_0` is more aggressive but has more noticeable impact)
- `OLLAMA_NUM_PARALLEL=1` — required memory scales with this multiplied by context length; keep it at 1 on constrained hardware
- `OLLAMA_MAX_LOADED_MODELS=1` — avoid multiple models resident in memory at once

A deeper dive into these settings, with benchmarks, is planned as a follow-up repo.

## Known limitations

- **Small dataset.** The current `raft_dataset.jsonl` has 6 records — a proof of concept, not a training-scale dataset.
- **Retrieval isn't perfect.** With `TOP_K=1`, retrieval occasionally pulls a chunk from the wrong source document. A `VERIFIED_FACTS` block in the generation prompt corrects the *content* of the answer when this happens, but the *cited source filename* can still be inaccurate in that case. Worth knowing if you inspect the dataset closely.
- **Small local models drift on formatting.** Even with explicit instructions, a small quantized model won't always follow output format exactly (e.g., citation placement) without a concrete example in the prompt. Providing an example, rather than just an instruction, measurably helped.
- **This repo does not include fine-tuning.** It stops at dataset generation. Actually fine-tuning a model on RAFT-style data is a larger, separate undertaking.

## License

MIT
