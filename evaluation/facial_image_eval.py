"""Run the bot's face observation pipeline on local photos without uploading them."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import core as main


async def evaluate(
    directory: Path, max_tokens: int, max_edge: int, only: set[str]
) -> None:
    main.FACE_VISION_MAX_TOKENS = max_tokens
    main.FACE_VISION_MAX_EDGE = max_edge
    completion = main.request_chat_completion
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (not only or path.name in only)
    )
    for path in paths:
        raw = None
        error = None

        def capture_completion(*args, **kwargs):
            nonlocal raw, error
            try:
                raw = completion(*args, **kwargs)
                return raw
            except RuntimeError as exc:
                error = str(exc)[:500]
                raise

        source = main.IncomingLesson(
            kind="image", url="https://localhost/local-evaluation",
            filename=path.name,
            mime_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg",
        )
        started = time.monotonic()
        with patch("lesson_content.download_lesson", return_value=path.read_bytes()), patch(
            "core.request_chat_completion", side_effect=capture_completion
        ):
            observation = await main.observe_face("local-evaluation", source)
        print(json.dumps({
            "file": path.name,
            "seconds": round(time.monotonic() - started, 2),
            "observation": vars(observation) if observation else None,
            "raw": raw[:500] if raw and observation is None else None,
            "error": error,
        }, ensure_ascii=True), flush=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-tokens", type=int, default=main.FACE_VISION_MAX_TOKENS)
    parser.add_argument("--max-edge", type=int, default=main.FACE_VISION_MAX_EDGE)
    parser.add_argument("--only", action="append", default=[], help="Evaluate one filename")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("Directory does not exist")
    if not main.LOCAL_VISION_MODEL:
        parser.error("Configure LOCAL_VISION_MODEL to run this local-only evaluation")
    asyncio.run(evaluate(args.directory, args.max_tokens, args.max_edge, set(args.only)))


if __name__ == "__main__":
    cli()
