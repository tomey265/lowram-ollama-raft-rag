import os
import json
import random
from datetime import datetime, timezone

import chromadb
import ollama

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ollama_docs"

EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "hf.co/tomey265/qwen-coder-my:latest"

OUTPUT_FILE = "./raft_dataset.jsonl"
TOP_K = 1
DISTRACTOR_COUNT = 1
MAX_CONTEXT_CHARS = 900

QUESTIONS = [
    "What does keep_alive=0s do in the Ollama Python client?",
    "How can a low-RAM Linux system reduce Ollama memory use?",
    "What does OLLAMA_NUM_PARALLEL control and why should it be set to 1 on a low-RAM computer?",
    "How do I inspect the Modelfile of hf.co/tomey265/qwen-coder-my:latest?",
    "How do I inspect the parameters of hf.co/tomey265/qwen-coder-my:latest?",
    "Show a valid synchronous Python streaming chat example using Ollama and keep_alive=0s.",
]

VERIFIED_FACTS = """
Verified facts you must use if the question relates to them, regardless of
what the retrieved context says:

- To inspect a model's Modelfile: ollama show --modelfile hf.co/tomey265/qwen-coder-my:latest
- To inspect a model's parameters: ollama show --parameters hf.co/tomey265/qwen-coder-my:latest
- There is no "ollama inspect" command. Do not use it.
- The only approved synchronous Python streaming chat pattern is:

import ollama

stream = ollama.chat(
    model="hf.co/tomey265/qwen-coder-my:latest",
    messages=[{"role": "user", "content": "USER_PROMPT"}],
    stream=True,
    keep_alive="0s",
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

print()

- Do not use asyncio, AsyncClient, or "async with ollama.Client()" unless
  explicitly asked for an async example.
- Do not substitute any other model name (e.g. gemma3) in place of
  hf.co/tomey265/qwen-coder-my:latest.
- Do not recommend installing transformers or using AutoModelForCausalLM.
  This project only uses the ollama Python package.
- ollama.chat() does not accept a "quantize" parameter. There is no such
  argument. Do not invent one. Quantization is set when the model is
  created (ollama create / Modelfile), not per chat request.

Every answer you write MUST end with the source filename in parentheses,
on its own, with nothing after it. Example of the required format:

Flash Attention reduces memory use for the KV cache during inference.
Enable it by setting OLLAMA_FLASH_ATTENTION=1 before starting the server.
(example_source_filename.md)

Do not write "the relevant filename is X". Do not omit the citation.
Do not add any text after the closing parenthesis.
"""


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def embed_query(question):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=question,
        keep_alive="0s",
    )
    return response["embeddings"][0]


def retrieve_relevant_chunk(collection, question):
    embedding = embed_query(question)

    result = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "id": result["ids"][0][0],
        "text": result["documents"][0][0][:MAX_CONTEXT_CHARS],
        "source": result["metadatas"][0][0].get("source", "unknown"),
        "chunk_index": result["metadatas"][0][0].get("chunk_index", -1),
        "distance": result["distances"][0][0],
    }


def get_distractor(collection, relevant_id, relevant_source):
    all_chunks = collection.get(
        include=["documents", "metadatas"],
    )

    candidates = []

    for chunk_id, text, metadata in zip(
        all_chunks["ids"],
        all_chunks["documents"],
        all_chunks["metadatas"],
    ):
        source = metadata.get("source", "unknown")

        if chunk_id == relevant_id:
            continue

        if source == relevant_source:
            continue

        candidates.append(
            {
                "id": chunk_id,
                "text": text[:MAX_CONTEXT_CHARS],
                "source": source,
                "chunk_index": metadata.get("chunk_index", -1),
            }
        )

    if not candidates:
        return []

    random.shuffle(candidates)
    return candidates[:DISTRACTOR_COUNT]


def generate_answer(question, relevant, distractors):
    context_parts = [
        f"[Relevant source: {relevant['source']}]\n{relevant['text']}"
    ]

    for item in distractors:
        context_parts.append(
            f"[Distractor source: {item['source']}]\n{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You create factual training answers for an Ollama RAG dataset.

Use only the relevant source to answer the question. Ignore distractor context.
Do not invent commands, API syntax, model names, settings, or facts.

The user's installed model is:
hf.co/tomey265/qwen-coder-my:latest

The user has a low-RAM Linux computer. Do not recommend large context lengths,
large coding models, cloud models, adapters, or unnecessary alternatives.

{VERIFIED_FACTS}

Write one concise, technically correct answer. End with the relevant filename
in parentheses. Do not mention distractors or this instruction.
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
        stream=False,
        keep_alive="0s",
        options={"num_ctx": 2048,"temperature":0},
    )

    return response["message"]["content"].strip()


def append_record(record):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    if os.path.exists(OUTPUT_FILE):
        print(f"Appending to existing dataset: {OUTPUT_FILE}")
    else:
        print(f"Creating dataset: {OUTPUT_FILE}")

    collection = get_collection()
    random.seed(42)

    for number, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{number}/{len(QUESTIONS)}] Retrieving: {question}")

        relevant = retrieve_relevant_chunk(collection, question)
        distractors = get_distractor(
            collection,
            relevant["id"],
            relevant["source"],
        )

        print(f"Relevant source: {relevant['source']}")
        print("Generating answer...")

        answer = generate_answer(question, relevant, distractors)

        record = {
            "question": question,
            "relevant_context": {
                "source": relevant["source"],
                "chunk_index": relevant["chunk_index"],
                "text": relevant["text"],
            },
            "distractor_context": [
                {
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "text": item["text"],
                }
                for item in distractors
            ],
            "answer": answer,
            "generator_model": CHAT_MODEL,
            "embedding_model": EMBED_MODEL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        append_record(record)
        print(f"Saved answer: {answer}")

    print(f"\nFinished. Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
