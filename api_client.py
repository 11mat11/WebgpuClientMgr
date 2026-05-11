from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import requests
import urllib3


@dataclass(frozen=True)
class ApiResponse:
    status: int
    json: dict[str, Any] | list[Any] | None
    client_rtt_ms: float


class SyncApiClient:
    def __init__(self, base_url: str, timeout_sec: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._timeout = timeout_sec
        self._verify_ssl = not self._base_url.lower().startswith("https://")
        if not self._verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
    ) -> ApiResponse:
        url = f"{self._base_url}{path}"
        start = time.perf_counter()
        response = self._session.request(
            method=method,
            url=url,
            json=json_body,
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return ApiResponse(status=response.status_code, json=payload, client_rtt_ms=elapsed_ms)

    def close(self) -> None:
        self._session.close()


class AsyncApiClient:
    def __init__(self, base_url: str, timeout_sec: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._session: aiohttp.ClientSession | None = None
        self._verify_ssl = not self._base_url.lower().startswith("https://")

    async def __aenter__(self) -> "AsyncApiClient":
        connector = None
        if not self._verify_ssl:
            connector = aiohttp.TCPConnector(ssl=False)
        self._session = aiohttp.ClientSession(timeout=self._timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
    ) -> ApiResponse:
        if self._session is None:
            raise RuntimeError("AsyncApiClient must be used as an async context manager.")
        url = f"{self._base_url}{path}"
        start = time.perf_counter()
        async with self._session.request(method=method, url=url, json=json_body) as response:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            try:
                payload = await response.json()
            except (aiohttp.ContentTypeError, ValueError):
                payload = None
        return ApiResponse(status=response.status, json=payload, client_rtt_ms=elapsed_ms)
