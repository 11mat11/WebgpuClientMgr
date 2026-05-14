from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

DEFAULT_SERVER_URL: Final[str] = "http://127.0.0.1:3000"


@dataclass(frozen=True)
class AppConfig:
    server_url: str
    use_cuda: bool


def load_config(force_no_cuda: bool = False) -> AppConfig:
    load_dotenv()
    server_url = os.getenv("SERVER_URL", DEFAULT_SERVER_URL).strip()
    use_cuda_env = os.getenv("USE_CUDA", "1").strip().lower()
    use_cuda = use_cuda_env in {"1", "true", "yes", "on"}
    if force_no_cuda:
        use_cuda = False
    return AppConfig(server_url=server_url, use_cuda=use_cuda)
