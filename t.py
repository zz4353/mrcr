import base64
import json
import sys
from pathlib import Path
from typing import Any

from mrcr_image_history import build_image_history_messages


DATA_PATH = Path("data/mini/val.jsonl")
OUTPUT_TXT = Path("runs/t_image_history_output.txt")
IMAGE_DIR = Path("runs/t_image_history_images")


def image_blocks(value: Any):
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                yield image_url["url"]
        for item in value.values():
            yield from image_blocks(item)
    elif isinstance(value, list):
        for item in value:
            yield from image_blocks(item)


def write_images(data_urls: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("*.png"):
        old_image.unlink()

    paths: list[Path] = []
    digits = len(str(max(1, len(data_urls))))
    for index, data_url in enumerate(data_urls, 1):
        if "," not in data_url:
            continue
        _, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        path = output_dir / f"page_{index:0{digits}d}.png"
        path.write_bytes(image_bytes)
        paths.append(path)
    return paths


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    row = json.loads(DATA_PATH.read_text(encoding="utf-8").splitlines()[0])
    messages = build_image_history_messages(
        row["messages"],
        recent_turns=3,
        highres_related_top_k=3,
    )

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

    data_urls = list(image_blocks(messages))
    image_paths = write_images(data_urls, IMAGE_DIR)

    print(f"input: {DATA_PATH}")
    print(f"txt: {OUTPUT_TXT}")
    print(f"images_dir: {IMAGE_DIR}")
    print(f"images: {len(image_paths)}")
    for path in image_paths:
        print(path)


if __name__ == "__main__":
    main()
