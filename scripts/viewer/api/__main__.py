from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "scripts.viewer.api.app:create_app",
        factory=True,
        host=os.environ.get("VIDEO_TO_3DGS_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("VIDEO_TO_3DGS_WEB_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
