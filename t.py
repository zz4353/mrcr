"""Convert the first val sample with image history and save the raw result."""

from __future__ import annotations

import json
from pathlib import Path

from mrcr_image_history import (
    ConversationImageRenderer,
    build_image_history_messages,
    split_recent_turns,
)


VAL_PATH = Path("data") / "mrcr" / "mini" / "val.jsonl"
OUTPUT_DIR = Path("debug_first_val_image_history")
OUTPUT_MESSAGES_PATH = OUTPUT_DIR / "converted_messages.txt"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
RECENT_TURNS = 3


def main() -> None:
    with VAL_PATH.open("r", encoding="utf-8") as file:
        row = json.loads(file.readline())

    renderer = ConversationImageRenderer()
    converted_messages = build_image_history_messages(
        row["messages"],
        recent_turns=RECENT_TURNS,
        renderer=renderer,
    )

    old_messages, _ = split_recent_turns(row["messages"], recent_turns=RECENT_TURNS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renderer.render_pages(
        old_messages,
        output_dir=OUTPUT_IMAGES_DIR,
        prefix="old_history",
        clear_output=True,
    )

    OUTPUT_MESSAGES_PATH.write_text(
        json.dumps(converted_messages, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
