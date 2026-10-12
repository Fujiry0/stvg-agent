"""Build Qwen-style messages for AnyGroundBench prompts."""
from typing import Any


def qwen_prepare_query(
    query_video_path: str,
    query_max_frames_num: int,
    fps: int,
    question_sample: str
) -> list[dict[str, Any]]:
    """Build a Qwen query message with video and text content."""
    return [
            {
                "role": "user",
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
                        "text": question_sample,
                    },
                ],
            }
        ]

def qwen_prepare_demonstration(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build Qwen demonstration messages with video, prompt, and answer."""
    return  [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": support_video_path,
                            # "max_pixels": 224 * 224,
                            "max_pixels": 384 * 384,
                            "max_frames": support_max_frames_num,
                            "min_frames": 1,
                            "fps": fps,
                        },
                        {
                            "type": "text",
                            "text": question_sample
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": str(ground_truth)
                        }
                    ]
                }
            ]


def qwen_prepare_demonstration_text_only(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build text-only Qwen demonstration messages."""
    return  [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question_sample
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": str(ground_truth)
                        }
                    ]
                }
            ]

def qwen_prepare_query_text_only(
    query_video_path: str,
    query_max_frames_num: int,
    fps: int,
    question_sample: str
) -> list[dict[str, Any]]:
    """Build a text-only Qwen query message."""
    return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question_sample,
                    },
                ],
            }
        ]
