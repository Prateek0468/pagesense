# PageSense

A book recommendation system: **Weaviate** vector store + Python backend for the
recommendation engine, and a **Next.js** frontend.

## Layout

```
.
├── docker-compose.yml   # Weaviate
├── backend/             # Python (FastAPI) recommendation engine
└── frontend/            # Next.js app
```

## Getting started

```bash
# Weaviate
docker compose up -d

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
pnpm install
pnpm dev
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Weaviate: http://localhost:8080
