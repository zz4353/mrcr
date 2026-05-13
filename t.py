"""Convert the first val sample with image history and save the raw result."""

from __future__ import annotations

import json
from pathlib import Path

from mrcr_image_history import build_image_history_messages


VAL_PATH = Path("data") / "mrcr" / "mini" / "val.jsonl"
OUTPUT_PATH = Path("data") / "mrcr" / "mini" / "first_val_image_history.txt"


def main() -> None:
    with VAL_PATH.open("r", encoding="utf-8") as file:
        row = json.loads(file.readline())

    converted_messages = build_image_history_messages(row["messages"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(converted_messages, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
