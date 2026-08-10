#!/usr/bin/env python3
"""Diagnose Kaggle / Jupyter public proxy routing for a local service port.

Run INSIDE the live Kaggle notebook session (with FastAPI already on :8000):

  python scripts/debug_kaggle_proxy.py
  python scripts/debug_kaggle_proxy.py --port 8000

Does NOT start services and does NOT modify application code.
Prints env metadata, candidate public URLs, and HTTP probe results so we can
see which path actually reaches the process listening on the given port.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
ENV_HINTS = ("KAGGLE", "JUPYTER", "PROXY", "PORT", "PUBLIC", "URL", "NB_", "SERVER")


def _log(msg: str) -> None:
    print(msg, flush=True)


def is_kaggle() -> bool:
    return bool(
        Path("/kaggle").exists()
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL")
    )


def list_jupyter_servers() -> List[Dict[str, Any]]:
    try:
        from jupyter_server.serverapp import list_running_servers
    except Exception as exc:
        _log(f"[warn] jupyter_server.list_running_servers unavailable: {exc}")
        return []
    try:
        return list(list_running_servers())
    except Exception as exc:
        _log(f"[warn] list_running_servers failed: {exc}")
        return []


def normalize_base(base: str) -> str:
    base = (base or "/").strip() or "/"
    if not base.startswith("/"):
        base = "/" + base
    if not base.endswith("/"):
        base += "/"
    return base


def pick_jupyter_base(servers: List[Dict[str, Any]]) -> Optional[str]:
    if not servers:
        return None
    ranked = sorted(
        servers,
        key=lambda s: (
            0 if "/k/" in str(s.get("base_url") or "") else 1,
            str(s.get("base_url") or ""),
        ),
    )
    base = str(ranked[0].get("base_url") or "").strip()
    return normalize_base(base) if base else None


def relevant_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        upper = key.upper()
        if any(h in upper for h in ENV_HINTS):
            # Redact long tokens but keep shape visible.
            shown = value
            if len(shown) > 120:
                shown = shown[:60] + f"...<{len(value)} chars>..." + shown[-20:]
            out[key] = shown
    return out


def local_probe(port: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {"port_open": False, "endpoints": {}}
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            result["port_open"] = True
    except OSError as exc:
        result["port_error"] = str(exc)
        return result

    for path in ("/", "/about", "/docs", "/api/v1/health", "/openapi.json"):
        url = f"http://127.0.0.1:{port}{path}"
        result["endpoints"][path] = probe_url(url, timeout=5.0)
    return result


def probe_url(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "url": url,
        "status": None,
        "location": None,
        "content_type": None,
        "server": None,
        "x_headers": {},
        "body_preview": "",
        "error": None,
    }
    req = Request(
        url,
        method="GET",
        headers={
            "User-Agent": "FabricVision-debug_kaggle_proxy/1.0",
            "Accept": "*/*",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(240)
            info["status"] = int(getattr(resp, "status", 200))
            info["location"] = resp.headers.get("Location")
            info["content_type"] = resp.headers.get("Content-Type")
            info["server"] = resp.headers.get("Server")
            info["x_headers"] = {
                k: v for k, v in resp.headers.items() if k.lower().startswith("x-")
            }
            info["body_preview"] = raw.decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw = b""
        try:
            raw = exc.read(240)
        except Exception:
            pass
        info["status"] = int(exc.code)
        info["location"] = exc.headers.get("Location") if exc.headers else None
        info["content_type"] = exc.headers.get("Content-Type") if exc.headers else None
        info["server"] = exc.headers.get("Server") if exc.headers else None
        if exc.headers:
            info["x_headers"] = {
                k: v for k, v in exc.headers.items() if k.lower().startswith("x-")
            }
        info["body_preview"] = raw.decode("utf-8", errors="replace")
        info["error"] = f"HTTPError: {exc}"
    except URLError as exc:
        info["error"] = f"URLError: {exc}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def looks_like_fastapi_or_next(probe: Dict[str, Any]) -> bool:
    """Heuristic: did this response come from our gateway, not a generic 404 page?"""
    status = probe.get("status")
    body = (probe.get("body_preview") or "").lower()
    ctype = (probe.get("content_type") or "").lower()
    if status == 200:
        if "fastapi" in body or "swagger" in body or "fabricvision" in body:
            return True
        if "text/html" in ctype and ("<!doctype html>" in body or "<html" in body):
            # Could be Next; treat 200 HTML as promising.
            return True
        if "application/json" in ctype and ("status" in body or "ok" in body):
            return True
    # Known non-app markers
    if status == 404 and ("page not found" in body or body.strip() == "404 page not found"):
        return False
    return False


def candidate_paths(jupyter_base: Optional[str], port: int) -> List[Tuple[str, str]]:
    """
    Return (label, path) candidates.

    Path theory under jupyter-server-proxy:
      public = {jupyter_base_url}proxy/{port}/
    If jupyter_base_url already ends with /proxy/, that yields .../proxy/proxy/{port}/.
    """
    port_s = str(int(port))
    cands: List[Tuple[str, str]] = [
        ("host_root_proxy_port", f"/proxy/{port_s}/"),
        ("host_root_proxy_port_nopslash", f"/proxy/{port_s}"),
    ]
    if jupyter_base:
        base = normalize_base(jupyter_base)
        trimmed = base.rstrip("/")
        # urljoin semantics (directory base + relative):
        via_urljoin = urljoin(base, f"proxy/{port_s}/")
        # Explicit segment joins:
        single = f"{trimmed}/{port_s}/"  # .../proxy/8000/
        double = f"{trimmed}/proxy/{port_s}/"  # .../proxy/proxy/8000/
        no_proxy_port = re.sub(r"/proxy$", f"/{port_s}", trimmed) + "/"
        # If base is /k/x/y/proxy → /k/x/y/8000/
        session_only = trimmed
        if session_only.endswith("/proxy"):
            session_only = session_only[: -len("/proxy")]
        session_port = f"{session_only}/{port_s}/" if session_only else f"/{port_s}/"

        cands.extend(
            [
                ("jupyter_base_as_is", base),
                ("append_port_only", single),
                ("append_proxy_port (double-proxy)", double),
                ("urljoin_proxy_port", via_urljoin if via_urljoin.endswith("/") else via_urljoin + "/"),
                ("replace_proxy_with_port", no_proxy_port),
                ("session_prefix_plus_port", session_port),
            ]
        )
    # De-dupe while preserving order
    seen = set()
    unique: List[Tuple[str, str]] = []
    for label, path in cands:
        if not path.startswith("/"):
            path = "/" + path
        key = path
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, path))
    return unique


def print_probe(label: str, probe: Dict[str, Any], mark: str = "") -> None:
    _log("-" * 72)
    _log(f"CANDIDATE: {label}{mark}")
    _log(f"  URL:           {probe.get('url')}")
    _log(f"  status:        {probe.get('status')}")
    _log(f"  Location:      {probe.get('location')}")
    _log(f"  Content-Type:  {probe.get('content_type')}")
    _log(f"  Server:        {probe.get('server')}")
    xh = probe.get("x_headers") or {}
    if xh:
        _log(f"  X-*:           {json.dumps(xh, ensure_ascii=False)}")
    if probe.get("error"):
        _log(f"  error:         {probe.get('error')}")
    preview = (probe.get("body_preview") or "").replace("\n", "\\n")
    _log(f"  body[:200]:    {preview[:200]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Kaggle public proxy paths")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default=os.environ.get("KAGGLE_JUPYTER_PROXY_HOST", DEFAULT_HOST).rstrip("/"),
        help="Public jupyter-proxy host",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override Jupyter base_url (otherwise auto-detect)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "experiments" / "generation_results" / "kaggle_proxy_debug.json"),
        help="Write full JSON report to this path",
    )
    args = parser.parse_args()

    _log("=" * 72)
    _log("FabricVision-AI Kaggle proxy diagnostics")
    _log("=" * 72)
    _log(f"cwd:          {os.getcwd()}")
    _log(f"is_kaggle:    {is_kaggle()}")
    _log(f"probe_host:   {args.host}")
    _log(f"local_port:   {args.port}")

    servers = list_jupyter_servers()
    _log("")
    _log("Jupyter list_running_servers():")
    if not servers:
        _log("  (none)")
    for i, srv in enumerate(servers):
        _log(
            f"  [{i}] base_url={srv.get('base_url')!r} "
            f"url={srv.get('url')!r} "
            f"hostname={srv.get('hostname')!r} "
            f"port={srv.get('port')!r} "
            f"pid={srv.get('pid')!r}"
        )

    jupyter_base = normalize_base(args.base_url) if args.base_url else pick_jupyter_base(servers)
    _log("")
    _log(f"Detected jupyter_base_url: {jupyter_base!r}")

    env = relevant_env()
    _log("")
    _log("Relevant environment variables:")
    if not env:
        _log("  (none matched)")
    for k, v in env.items():
        _log(f"  {k}={v}")

    _log("")
    _log("Local service probe (127.0.0.1):")
    local = local_probe(args.port)
    _log(f"  port_open={local.get('port_open')}")
    for path, probe in (local.get("endpoints") or {}).items():
        _log(f"  {path} -> status={probe.get('status')} ctype={probe.get('content_type')}")

    paths = candidate_paths(jupyter_base, args.port)
    _log("")
    _log(f"Probing {len(paths)} public candidate URL(s)...")

    results: List[Dict[str, Any]] = []
    hits: List[str] = []
    for label, path in paths:
        # Root + a couple of app endpoints for promising paths.
        root_url = f"{args.host}{path}"
        root_probe = probe_url(root_url)
        promising = looks_like_fastapi_or_next(root_probe)
        mark = "  << POSSIBLE HIT" if promising else ""
        print_probe(label, root_probe, mark=mark)
        entry = {"label": label, "path": path, "root": root_probe, "extra": {}}
        if promising or root_probe.get("status") in {200, 301, 302, 307, 308}:
            for suffix, name in (
                ("about", "about"),
                ("docs", "docs"),
                ("api/v1/health", "health"),
            ):
                # Join carefully onto directory-style path.
                base_path = path if path.endswith("/") else path + "/"
                extra_url = f"{args.host}{base_path}{suffix}"
                extra_probe = probe_url(extra_url)
                entry["extra"][name] = extra_probe
                print_probe(f"{label} + {suffix}", extra_probe)
                if looks_like_fastapi_or_next(extra_probe):
                    promising = True
        if promising:
            hits.append(label)
        results.append(entry)

    _log("")
    _log("=" * 72)
    _log("SUMMARY")
    _log("=" * 72)
    _log(f"jupyter_base_url: {jupyter_base!r}")
    if hits:
        _log("Candidates that appear to reach the local app:")
        for h in hits:
            _log(f"  - {h}")
    else:
        _log(
            "No candidate clearly reached FastAPI/Next from this host. "
            "Common causes: wrong path shape, missing notebook auth cookie/token "
            "when probing the public host from inside the VM, or Kaggle edge routing "
            "that only works from an authenticated browser session."
        )
        _log(
            "Also open the top candidates manually in the browser while logged into "
            "the same Kaggle session, and compare."
        )

    # Theory reminder for operators reading the log.
    if jupyter_base and normalize_base(jupyter_base).rstrip("/").endswith("/proxy"):
        expected_jsp = normalize_base(jupyter_base).rstrip("/") + f"/proxy/{args.port}/"
        _log("")
        _log("jupyter-server-proxy theory (base_url + 'proxy/<port>/'):")
        _log(f"  expected path = {expected_jsp}")
        _log(
            "  Note: the first '/proxy/' is Kaggle's Jupyter tunnel prefix; "
            "the second '/proxy/<port>/' is jupyter-server-proxy's port mapper. "
            "That 'double proxy' is often intentional, not a string bug."
        )

    report = {
        "is_kaggle": is_kaggle(),
        "host": args.host,
        "port": args.port,
        "jupyter_base_url": jupyter_base,
        "servers": servers,
        "env": env,
        "local": local,
        "results": results,
        "hits": hits,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log("")
    _log(f"Wrote JSON report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
