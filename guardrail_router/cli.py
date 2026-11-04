"""Command line entry point: serve the API or screen a file of samples."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .app import create_app
from .config import get_settings
from .pipeline import ScreeningPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guardrail-serve",
        description="Screen untrusted text before it reaches a downstream model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true", help="reload on code changes")

    screen = sub.add_parser("screen", help="screen a JSONL file of texts")
    screen.add_argument("--file", type=Path, required=True)
    screen.add_argument("--tenant", default="cli")
    screen.add_argument("--json", action="store_true", help="print one JSON verdict per line")
    return parser


def _read_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"{path}:{number}: missing a non-empty 'text' field")
            texts.append(text)
    return texts


async def _screen_file(path: Path, tenant: str, as_json: bool) -> int:
    settings = get_settings()
    pipeline = ScreeningPipeline.from_settings(settings)
    texts = _read_texts(path)
    if not texts:
        print(f"no samples found in {path}", file=sys.stderr)
        return 1

    blocked = 0
    for text in texts:
        verdict = await pipeline.screen(text, context=None)
        if verdict.decision == "block":
            blocked += 1
        if as_json:
            print(json.dumps({"tenant_id": tenant, **verdict.model_dump()}))
        else:
            print(
                f"{verdict.decision:<6} {verdict.category:<12} "
                f"score={verdict.score:.3f} stage={verdict.stage:<10} "
                f"{verdict.latency_ms:6.1f}ms  {verdict.excerpt}"
            )

    print(
        f"\n{len(texts)} screened, {blocked} blocked, "
        f"{len(texts) - blocked} allowed or queued for review",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "serve":
        import uvicorn

        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "screen":
        return asyncio.run(_screen_file(args.file, args.tenant, args.json))

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
