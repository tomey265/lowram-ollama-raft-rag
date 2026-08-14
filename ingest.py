import hashlib
import os
import shutil
import sys

import chromadb
import ollama

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ollama_docs"
EMBED_MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 16

RESET_COLLECTION = False


def read_documents():
    documents = []

    if not os.path.isdir(DOCS_DIR):
        print(f"Error: {DOCS_DIR} does not exist. Run fetch_docs.py first.")
        sys.exit(1)

    for root, _, files in os.walk(DOCS_DIR):
        for filename in sorted(files):
            if not filename.lower().endswith((".md", ".txt")):
                continue

            path = os.path.join(root, filename)

            try:
                with open(path, "r", encoding="utf-8", errors="replace") as file:
                    text = file.read().strip()
            except OSError as error:
                print(f"Skipping {path}: {error}")
                continue

            if text:
                documents.append((path, text))

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            break_points = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best_break = max(break_points)

            if best_break > start + (chunk_size // 2):
                end = best_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def make_chunk_id(source, chunk_index, text):
    value = f"{source}:{chunk_index}:{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_collection():
    if RESET_COLLECTION and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Official Ollama documentation chunks"},
    )


def embed_texts(texts):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
        keep_alive="0s",
    )
    return response["embeddings"]


def ingest_documents(collection, documents):
    ids = []
    texts = []
    metadatas = []

    for source_path, document_text in documents:
        source_name = os.path.relpath(source_path, DOCS_DIR)
        chunks = chunk_text(document_text)

        print(f"Chunking {source_name}: {len(chunks)} chunks")

        for chunk_index, chunk in enumerate(chunks):
            ids.append(make_chunk_id(source_name, chunk_index, chunk))
            texts.append(chunk)
            metadatas.append(
                {
                    "source": source_name,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk),
                }
            )

    if not texts:
        print("No usable documentation chunks found.")
        return

    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(texts), EMBED_BATCH_SIZE),
        start=1,
    ):
        end = start + EMBED_BATCH_SIZE
        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        batch_metadata = metadatas[start:end]

        print(
            f"Embedding batch {batch_number}/{total_batches} "
            f"({len(batch_texts)} chunks)..."
        )

        embeddings = embed_texts(batch_texts)

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadata,
        )

    print(f"\nIngestion complete. Collection now contains {collection.count()} chunks.")


def main():
    documents = read_documents()

    if not documents:
        print("No .md or .txt files found in ./docs.")
        sys.exit(1)

    print(f"Found {len(documents)} documentation files.")
    collection = get_collection()
    ingest_documents(collection, documents)


if __name__ == "__main__":
    main()
