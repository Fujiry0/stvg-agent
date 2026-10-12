"""Build Eagle messages for AnyGroundBench prompts."""
from typing import Any


def eagle_prepare_query(
    query_video_path: str,
    query_max_frames_num: int,
    fps: int,
    question_sample: str,
) -> list[dict[str, Any]]:
    """Build an Eagle query message with video and text content."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": query_video_path,
                    "max_frames": query_max_frames_num,
                    "min_frames": 1,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": question_sample,
                },
            ],
        }
    ]


def eagle_prepare_demonstration(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build Eagle demonstration messages with video, prompt, and answer."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": support_video_path,
                    "max_frames": support_max_frames_num,
                    "min_frames": 1,
                    "fps": fps,
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


def eagle_prepare_demonstration_text_only(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build text-only Eagle demonstration messages."""
    del support_video_path, support_max_frames_num, fps
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question_sample,
                }
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
