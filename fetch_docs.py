import os
import urllib.request
import urllib.error

DOCS_DIR = "./docs"

DOCUMENTS = {
    "qwen_coder_official.md": (
        "https://raw.githubusercontent.com/QwenLM/Qwen3-Coder/main/README.md"
    ),
    "ollama_python.md": (
        "https://raw.githubusercontent.com/ollama/ollama-python/main/README.md"
    )}

def download_file(filename, url):
    destination = os.path.join(DOCS_DIR, filename)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ollama-rag-doc-fetcher/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()

        with open(destination, "wb") as file:
            file.write(content)

        print(f"Downloaded: {filename} ({len(content):,} bytes)")
        return True

    except urllib.error.HTTPError as error:
        print(f"FAILED: {filename} - HTTP {error.code}: {error.reason}")
    except urllib.error.URLError as error:
        print(f"FAILED: {filename} - Network error: {error.reason}")
    except OSError as error:
        print(f"FAILED: {filename} - File error: {error}")

    return False


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    succeeded = 0
    for filename, url in DOCUMENTS.items():
        if download_file(filename, url):
            succeeded += 1

    print(f"\nFinished: {succeeded}/{len(DOCUMENTS)} files downloaded to {DOCS_DIR}")


if __name__ == "__main__":
    main()
