"""Helpers for serving the Chainlit UI."""

from __future__ import annotations

import os
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

CHAINLIT_APP_PATH_ENV = "CV_SCREENER_CHAINLIT_APP_PATH"
CHAINLIT_MOUNT_PATH = "/chainlit"


@dataclass(frozen=True)
class ChainlitServeRequest:
    """Options for serving the Chainlit application."""

    app_path: Path
    host: str
    port: int
    headless: bool
    watch: bool


class ChainlitLauncher:
    """Serve Chainlit by mounting it into a lightweight FastAPI host app."""

    def run(self, request: ChainlitServeRequest) -> int:
        """Run the host app and return 0 when the server exits cleanly."""
        os.environ[CHAINLIT_APP_PATH_ENV] = str(request.app_path)
        if not request.headless:
            webbrowser.open(_chainlit_url(request.host, request.port))

        uvicorn.run(
            "cv_screener.cli.serve:create_chainlit_host_app",
            factory=True,
            host=request.host,
            port=request.port,
            reload=request.watch,
            reload_dirs=[str(_project_root())] if request.watch else None,
        )
        return 0


def create_chainlit_host_app() -> object:
    """Create a FastAPI app with the project Chainlit UI mounted under `/chainlit`."""
    app = FastAPI()

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url=CHAINLIT_MOUNT_PATH)

    mount_chainlit(
        app=app,
        target=str(_chainlit_app_path_from_env()),
        path=CHAINLIT_MOUNT_PATH,
    )
    return app


def default_chainlit_app_path() -> Path:
    """Return the absolute path to the project Chainlit app module."""
    return Path(__file__).resolve().parents[1] / "chainlit" / "app.py"


def _project_root() -> Path:
    """Return the repository root for subprocess execution."""
    return Path(__file__).resolve().parents[3]


def _chainlit_app_path_from_env() -> Path:
    """Return the Chainlit target path configured for the current serve session."""
    configured_path = os.environ.get(CHAINLIT_APP_PATH_ENV)
    if configured_path:
        return Path(configured_path)
    return default_chainlit_app_path()


def _chainlit_url(host: str, port: int) -> str:
    """Return the local browser URL for the mounted Chainlit UI."""
    return f"http://{host}:{port}{CHAINLIT_MOUNT_PATH}"
