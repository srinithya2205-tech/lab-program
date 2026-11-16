"""FastAPI app for the Portfolio Intelligence Assistant.

Run locally:      uvicorn app.main:app --reload --port 8000
Run in production: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
"""

import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .engine import PortfolioAssistant, load_portfolio
from .schemas import AskRequest, AskResponse, HealthResponse

SERVICE_NAME = "portfolio-intelligence-assistant"
VERSION = "1.0.0"

# Browsers send only the ORIGIN (scheme + host [+ port]) in the Origin header,
# never the path. "https://srinithya2205-tech.github.io" covers the whole
# GitHub Pages site, including the /lab-program/ path.
DEFAULT_ORIGINS = [
    "https://srinithya2205-tech.github.io",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

logger = logging.getLogger("portfolio_assistant")


def allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or DEFAULT_ORIGINS


def create_app() -> FastAPI:
    portfolio = load_portfolio()
    assistant = PortfolioAssistant(portfolio)

    app = FastAPI(
        title="Portfolio Intelligence Assistant API",
        version=VERSION,
        description="Answers questions about the portfolio using only the data in data/portfolio.json.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=VERSION,
            portfolio_loaded=bool(portfolio.get("projects")),
        )

    @app.get("/portfolio")
    def get_portfolio() -> dict:
        return portfolio

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        try:
            result = assistant.ask(payload.question)
        except Exception:  # noqa: BLE001 - never leak internals to the client
            logger.exception("Assistant failed to answer a question")
            raise HTTPException(status_code=500, detail="The assistant hit an unexpected error.")
        return AskResponse(**result)

    return app


app = create_app()
