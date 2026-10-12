"""Build LLaVA-ST messages for AnyGroundBench prompts."""
from typing import Any


def llava_st_prepare_query(
    query_video_path: str,
    query_max_frames_num: int,
    fps: int,
    question_sample: str,
) -> list[dict[str, Any]]:
    """Build a LLaVA-ST query message with native and shared content fields."""
    # Keep llava_st-native keys (from/value/video_path/max_frames) for model input,
    # and also provide qwen-style content for shared frame selection utilities.
    return [
        {
            "from": "human",
            "value": f"<video>\n{question_sample.strip()}",
            "video_path": query_video_path,
            "max_frames": query_max_frames_num,
            "content": [
                {
                    "type": "video",
                    "video": query_video_path,
                    "max_pixels": 384 * 384,
                    "max_frames": query_max_frames_num,
                    "min_frames": 1,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": question_sample.strip(),
                },
            ],
        },
    ]


def llava_st_prepare_demonstration(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build LLaVA-ST demonstration messages with prompt and answer."""
    del support_video_path, support_max_frames_num, fps
    return [
        {
            "from": "human",
            "value": f"<video>\n{question_sample.strip()}",
        },
        {
            "from": "gpt",
            "value": str(ground_truth),
        },
    ]


def llava_st_prepare_demonstration_text_only(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build text-only LLaVA-ST demonstration messages."""
    del support_video_path, support_max_frames_num, fps
    return [
        {
            "from": "human",
            "value": question_sample,
        },
        {
            "from": "gpt",
            "value": str(ground_truth),
        },
    ]


def llava_st_to_conversations(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str, int]:
    """Convert LLaVA-ST messages into conversations and query video metadata."""
    conversations: list[dict[str, str]] = []
    query_video_path: str | None = None
    query_max_frames: int = 100

    for message in messages:
        from_role = message.get("from")
        value = str(message.get("value", "")).strip()
        if from_role in {"human", "gpt"}:
            conversations.append({"from": from_role, "value": value})
        if from_role == "human" and query_video_path is None:
            video_path = message.get("video_path")
            if isinstance(video_path, str) and video_path:
                query_video_path = video_path
                max_frames = message.get("max_frames")
                if isinstance(max_frames, int):
                    query_max_frames = max_frames

    if query_video_path is None:
        raise ValueError("llava_st requires query message with video_path")

    print("[llava_st] Input prompt:")
    print(f"[llava_st] query_video_path={query_video_path}")
    for idx, conv in enumerate(conversations):
        print(f"[llava_st] {idx:02d} {conv['from']}: {conv['value']}")

    return conversations, query_video_path, query_max_frames
