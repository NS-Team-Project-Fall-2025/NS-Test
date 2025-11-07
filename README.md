# NetSec Tutor

NetSec Tutor is an AI-assisted learning environment for Network Security Course. It pairs a Django REST API with a Next.js frontend to deliver a personalised tutor chat experience, personalised knowledge base, and adaptive quizzes powered by an Ollama-hosted large language model.

## Features

- AI tutor chat with streaming responses grounded in your knowledge base.
- Knowledge Base dashboard for uploading textbooks or lecture slides and rebuilding embeddings on demand.
- Quiz Lab that generates multi-format quizzes, grades submissions, and stores detailed attempt history.
- Session management to resume tutor conversations across devices.
- REST API surface suitable for integrating custom automations or dashboards.

## Tech Stack

- Backend: Django 4, LangChain, ChromaDB, sentence-transformers.
- Frontend: Next.js 14, React 18.
- LLM Runtime: Ollama (default `mistral` model, configurable).

## Repository Layout

- `backend/` – Django project (`netsectutor`) contains tutor agent, quiz agent, knowledge base, and session endpoints.
- `frontend/` – Next.js application for Tutor, Knowledge Base, Quiz Lab, and Sessions pages.
- `data/` – Knowledge base directories (`textbooks/`, `slides/`, and quiz history).
- `vectorstore/` – Persisted ChromaDB collections keyed by knowledge source.
- `config.py` – Shared configuration helpers and environment variable defaults.

## Prerequisites

- Python 3.10 or newer.
- Node.js 18.17 or newer and npm.
- [Ollama](https://ollama.ai) running locally with the desired model pulled (`ollama pull mistral` by default).
- macOS, Linux, or WSL environment with build tooling for Python wheels.

## Quick Start

### 1. Prepare the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start Ollama in another terminal:

```bash
ollama serve
ollama pull mistral # or any model defined via OLLAMA_MODEL (First time)
ollama run mistral  # run the model directly
```

### 2. Run the Django backend

```bash
python backend/manage.py migrate      # first-time setup
python backend/manage.py runserver 0.0.0.0:8000
```

The API base URL defaults to `http://localhost:8000/api/`.

### 3. Run the Next.js frontend

```bash
cd frontend
npm install
npm run dev
```

By default the app is served on `http://localhost:3000`, pointing to the backend API.

## Environment Variables

The project reads configuration from `config.py`. The configurable variables are:

| Variable | Description | Default |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | URL of the Ollama instance | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name to use for chat and quiz generation | `mistral` |
| `OLLAMA_TEMPERATURE` | Sampling temperature for LLM calls | `0.7` |
| `VECTOR_STORE_TYPE` | Embedding store backend | `chromadb` |
| `EMBEDDING_MODEL` | Sentence-transformer model for embeddings | `all-MiniLM-L6-v2` |
| `NETSEC_DATA_DIR` | Root directory for chat sessions and quiz history | `<project>/data/documents` |
| `NETSEC_VECTOR_DIR` | Root directory for vector stores | `<project>/vectorstore` |
| `NETSEC_DATA_TEXTBOOKS` | Textbook source directory | `<project>/data/textbooks` |
| `NETSEC_DATA_SLIDES` | Lecture slide source directory | `<project>/data/slides` |
| `NETSEC_VECTOR_TEXTBOOKS` | Persisted vectors for textbooks | `<project>/vectorstore/textbooks` |
| `NETSEC_VECTOR_SLIDES` | Persisted vectors for slides | `<project>/vectorstore/slides` |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend override for API base URL | `http://localhost:8000/api` |

## Working with the Knowledge Base

- Place organised source material under `data/textbooks/` or `data/slides/`.
- Use the Knowledge Base page in the UI to upload new files or trigger a rebuild of embeddings. Rebuilds refresh the vector store and inform quiz generation automatically.
- Persisted embeddings live under `vectorstore/` and can be backed up as needed.

## API Highlights

- `GET /api/config/` – Global metadata and runtime settings.
- `POST /api/chat/stream/` – Streaming tutor responses (NDJSON).
- `GET|POST /api/kb/...` – Knowledge base management endpoints for listing, uploading, and rebuilding documents.
- `POST /api/quiz/generate/` – Create a new adaptive quiz.
- `POST /api/quiz/grade/` – Grade quiz answers and record attempt history.
- `GET /api/sessions/` – Fetch tutor chat sessions and stored transcripts.

## Development Notes

- The Tutor page only persists messages for the active session; create or switch sessions from the dropdown before chatting.
- Quiz Lab supports topic filters, difficulty levels, and source selection (textbooks, slides, or both). Completed attempts are stored in `data/quiz_history/`.
- The frontend expects the backend to stream chat tokens; keep both servers running for the full experience.
- When adjusting configuration, restart the Django server to ensure new environment variables are loaded.



## Troubleshooting

- **LLM errors** – Confirm Ollama is running and the model listed in `OLLAMA_MODEL` is installed.
- **Embedding rebuild failures** – Check file permissions within `data/` and ensure the Python process has access to GPU/CPU resources needed by sentence-transformers.
- **API 404/500** – Inspect the Django console output and visit `http://localhost:8000/api/config/` to confirm the backend is reachable.
