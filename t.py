import json
from pathlib import Path

from mrcr_image_history.search import content_to_text, join_role_text
from mrcr_image_history.search2 import search_related_turns


def preview(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


row = json.loads(Path("data/mini/val.jsonl").read_text(encoding="utf-8").splitlines()[1])
messages = row["messages"]
active_user = content_to_text(messages[-1].get("content", ""))
related = search_related_turns(messages, active_user, top_k=5)

print(f"user_input: {active_user}")
print(f"expected desired_msg_index: {row.get('desired_msg_index')}")
print(f"answer_prefix: {preview(row.get('answer', ''), 160)}")
print()

for rank, item in enumerate(related, 1):
    user_text = join_role_text(item.turn.messages, role="user")
    assistant_text = join_role_text(item.turn.messages, role_not="user")
    print(
        f"#{rank} score={item.score:.3f} "
        f"turn={item.turn.turn_index} "
        f"messages={item.turn.start_message_index + 1}-{item.turn.end_message_index + 1} "
        f"reasons={', '.join(item.reasons)}"
    )
    print(f"user: {preview(user_text)}")
    print(f"assistant: {preview(assistant_text, 320)}")
    print()
