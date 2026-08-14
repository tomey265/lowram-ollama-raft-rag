> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Quickstart

Install Ollama and get your first response.

## 1. Download Ollama

Ollama runs on macOS, Windows, and Linux.

<a href="https://ollama.com/download" target="_blank" className="inline-block px-6 py-2 bg-black rounded-full dark:bg-neutral-700 text-white font-normal border-none">
  Download Ollama
</a>

## 2. Open the menu

Run `ollama` in your terminal to open the interactive menu:

```shell theme={"system"}
ollama
```

From the menu you can:

* **Run a model** - Start an interactive chat
* **Launch tools** - [Claude Code](/integrations/claude-code), [OpenClaw](/integrations/openclaw), [VS Code](/integrations/vscode), and more

## 3. Start a chat

Run a model to start your first chat.

```shell theme={"system"}
ollama run gemma4
```

Cloud models work the same way:

```shell theme={"system"}
ollama run gemma4:cloud
```

Send your first message:

```text theme={"system"}
Explain why the sky is blue in one paragraph.
```

To leave the chat, type:

```shell theme={"system"}
/bye
```

## Next steps

Use a model with an [integration](/integrations), make an [API request](/api/introduction), or browse more [models](https://ollama.com/search).
