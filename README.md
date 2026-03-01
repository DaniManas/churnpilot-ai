# ChurnPilot AI

Production-ready ML platform for telecom customer churn prediction with FastAPI APIs, async batch scoring, drift monitoring, and a Streamlit dashboard with a RAG-powered assistant.

## Features
- Single-customer churn inference (`POST /predict`)
- Async CSV batch scoring with job tracking (`POST /batch`)
- Model monitoring with PSI drift metrics (`GET /monitor`)
- RAG assistant over project knowledge docs (`POST /assist`)
- API key authentication and request logging
- Streamlit UI for prediction, batch, monitoring, and assistant workflows
- Docker + docker-compose setup

## Tech Stack
- FastAPI, SQLAlchemy, Pydantic, Uvicorn
- scikit-learn, pandas, numpy
- FAISS + optional OpenAI (`gpt-4o-mini`)
- Streamlit + Plotly
- pytest

## Quickstart
```bash
uv pip install -r requirements.txt
uv pip install streamlit plotly requests
cp .env.example .env
uv run uvicorn app.main:app --reload
uv run streamlit run streamlit_app.py
```

## Endpoints
- `GET /health`
- `POST /predict`
- `POST /batch`
- `GET /batch/{job_id}`
- `GET /batch/{job_id}/download`
- `GET /monitor`
- `POST /assist`

## Test
```bash
uv run pytest -q
```

## Notes
- `/predict` uses local sklearn model.
- `/assist` uses OpenAI only when `LLM_PROVIDER=openai`.
- Keep secrets in `.env`; never commit keys.
