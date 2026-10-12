"""Build GPT messages for AnyGroundBench prompts."""
from typing import Any, Callable


def gpt_prepare_query(
    query_video_path: str,
    question_sample: str,
    fps: int | None = None,
    query_max_frames_num: int | None = None,
) -> list[dict[str, Any]]:
    """Build a GPT query message with video and text content."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": query_video_path,
                    "fps": fps,
                    "max_frames": query_max_frames_num,
                },
                {
                    "type": "text",
                    "text": question_sample,
                },
            ],
        }
    ]


def gpt_prepare_demonstration(
    support_video_path: str,
    question_sample: str,
    ground_truth: list | str,
    fps: int | None,
    support_max_frames_num: int | None = None,
) -> list[dict[str, Any]]:
    """Build GPT demonstration messages with video, prompt, and answer."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": support_video_path,
                    "fps": fps,
                    "max_frames": support_max_frames_num,
                },
                {
                    "type": "text",
                    "text": question_sample,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": str(ground_truth),
                }
            ],
        },
    ]


def normalize_gpt_message_for_responses(
    message: dict[str, Any],
    video_to_image_urls: Callable[[dict[str, Any]], list[str]] | None = None,
) -> dict[str, Any]:
    """Convert the custom message format into Responses API input format."""
    role = message.get("role", "user")
    content = message.get("content", "")
    text_type = "output_text" if role == "assistant" else "input_text"

    if isinstance(content, str):
        return {
            "role": role,
            "content": [{"type": text_type, "text": content}],
        }

    if not isinstance(content, list):
        return {
            "role": role,
            "content": [{"type": text_type, "text": str(content)}],
        }

    normalized_content: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            normalized_content.append({"type": text_type, "text": str(part)})
            continue

        part_type = part.get("type")
        if part_type == "text":
            normalized_content.append(
                {"type": text_type, "text": str(part.get("text", ""))}
            )
            continue

        if part_type == "video":
            video_path = part.get("video")
            if not isinstance(video_path, str):
                continue

            if video_to_image_urls is None:
                continue
            image_urls = video_to_image_urls(part)
            normalized_content.extend(
                [{"type": "input_image", "image_url": url} for url in image_urls]
            )
            continue

        if part_type in {"input_text", "input_image", "output_text"}:
            normalized_content.append(part)
            continue

        normalized_content.append({"type": text_type, "text": str(part)})

    return {"role": role, "content": normalized_content}
