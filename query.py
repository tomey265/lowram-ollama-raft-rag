import sys

import chromadb
import ollama

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ollama_docs"

EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "hf.co/tomey265/qwen-coder-my:latest"

TOP_K = 4
NUM_CTX = 2048


def get_prompt():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    print("Ask a question about Ollama. Press Ctrl+D to finish:\n")

    try:
        prompt = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print()
        sys.exit(0)

    return prompt


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(
            f"Could not find the '{COLLECTION_NAME}' collection in {CHROMA_DIR}.\n"
            "Run ingest.py first."
        )
        sys.exit(1)


def embed_query(prompt):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=prompt,
        keep_alive="0s",
    )
    return response["embeddings"][0]


def retrieve_context(collection, prompt):
    query_embedding = embed_query(prompt)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        sources.append(f"{source} (chunk {chunk_index}, distance {distance:.4f})")

        context_parts.append(
            f"[Source {index}: {source}, chunk {chunk_index}]\n{document}"
        )

    return "\n\n---\n\n".join(context_parts), sources


def build_messages(question, context):
	system_prompt = """You are an Ollama documentation assistant for a user with
approximately 13 GiB total RAM, limited available memory, and active swap use.

Answer only from the supplied documentation context. If the context does not
contain enough information, say so clearly instead of guessing.

The user runs this installed coding model:
hf.co/tomey265/qwen-coder-my:latest

Prioritize low-RAM local Linux use. Never recommend a larger context length as
a way to reduce memory use. For this user, prefer small context settings such
as 2048 unless the retrieved documentation gives a specific reason otherwise.

Do not recommend Qwen3-Coder-Next, Qwen3-Coder 480B, cloud models, adapters,
or generic model substitutions unless the user explicitly asks about them.

When asked how to inspect the installed Qwen model, use these exact commands:
ollama show hf.co/tomey265/qwen-coder-my:latest
ollama show --modelfile hf.co/tomey265/qwen-coder-my:latest
ollama show --parameters hf.co/tomey265/qwen-coder-my:latest

Use examples with the user's installed model when a model name is needed.
Cite actual retrieved filenames in parentheses. Do not use labels such as
"Source 1" or "Source 2".For Python examples using the Ollama library, never invent async context-manager
syntax and never use .read() on a chat response.

Use this exact synchronous streaming pattern when the user asks for a Python
streaming chat example:

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

Do not substitute another model name. Do not use asyncio unless the user
specifically requests an asynchronous example.
"""
	user_prompt = f"""Documentation context:

{context}

Question:
{question}
"""

	return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def stream_answer(messages):
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        keep_alive="0s",
        options={
            "num_ctx": NUM_CTX,
        },
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)

    print()


def main():
    question = get_prompt()

    if not question:
        print("Usage: python query.py \"Your Ollama question here\"")
        sys.exit(1)

    collection = get_collection()
    context, sources = retrieve_context(collection, question)

    if not context:
        print("No relevant documentation was retrieved.")
        sys.exit(1)

    print("\nRetrieved documentation chunks:")
    for source in sources:
        print(f"- {source}")

    print("\nAnswer:\n")
    stream_answer(build_messages(question, context))


if __name__ == "__main__":
    main()
