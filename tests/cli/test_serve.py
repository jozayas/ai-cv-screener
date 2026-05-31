import os
from pathlib import Path

import pytest

import cv_screener.cli.serve as serve_module
from cv_screener.cli.serve import (
    CHAINLIT_APP_PATH_ENV,
    CHAINLIT_MOUNT_PATH,
    ChainlitLauncher,
    ChainlitServeRequest,
    _chainlit_url,
    default_chainlit_app_path,
)


def test_default_chainlit_app_path_points_to_project_app() -> None:
    assert default_chainlit_app_path() == Path(
        "/home/jozayas/Leadtech/cv-screener/src/cv_screener/chainlit/app.py"
    )


def test_chainlit_launcher_runs_uvicorn_with_factory_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    app_path = tmp_path / "chainlit_app.py"
    monkeypatch.setenv(CHAINLIT_APP_PATH_ENV, "")

    def fake_run(app: str, **kwargs: object) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(serve_module.uvicorn, "run", fake_run)

    exit_code = ChainlitLauncher().run(
        ChainlitServeRequest(
            app_path=app_path,
            host="127.0.0.1",
            port=9000,
            headless=True,
            watch=True,
        )
    )

    assert exit_code == 0
    assert captured["app"] == "cv_screener.cli.serve:create_chainlit_host_app"
    assert captured["kwargs"] == {
        "factory": True,
        "host": "127.0.0.1",
        "port": 9000,
        "reload": True,
        "reload_dirs": ["/home/jozayas/Leadtech/cv-screener"],
    }
    assert app_path == Path(os.environ[CHAINLIT_APP_PATH_ENV])


def test_chainlit_url_points_to_mounted_app() -> None:
    assert _chainlit_url("127.0.0.1", 8000) == f"http://127.0.0.1:8000{CHAINLIT_MOUNT_PATH}"
