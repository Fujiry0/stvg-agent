"""Build Gemini messages for AnyGroundBench prompts."""
from typing import Any, Callable

from google.genai import types


def gemini_prepare_query(
    query_video_path: str,
    question_sample: str,
    fps: int,
    query_max_frames_num=None,
) -> list[dict[str, Any]]:
    """Build a Gemini query message with video and text content."""
    del query_max_frames_num

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": query_video_path,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": question_sample,
                },
            ],
        }
    ]

def gemini_prepare_demonstration(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build Gemini demonstration messages with video, prompt, and answer."""
    del support_max_frames_num

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": support_video_path,
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
                },
            ],
        },
    ]

def gemini_prepare_query_text_only(
    query_video_path: str,
    question_sample: str,
    fps: int,
    query_max_frames_num=None,
) -> list[dict[str, Any]]:
    """Build a text-only Gemini query message."""
    del query_max_frames_num

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

def gemini_prepare_demonstration_text_only(
    support_video_path: str,
    support_max_frames_num: int,
    fps: int,
    question_sample: str,
    ground_truth: str,
) -> list[dict[str, Any]]:
    """Build text-only Gemini demonstration messages."""
    del support_max_frames_num

    return [
        {
            "role": "user",
            "content": [
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
                },
            ],
        },
    ]


def normalize_gemini_messages(
    messages: list[dict[str, Any]],
    upload_video: Callable[[str], Any],
    default_fps: int | None = None,
    uploaded_file_names_out: list[str] | None = None,
) -> list[types.Content]:
    """Convert custom messages into Gemini SDK content objects."""
    inputs: list[types.Content] = []

    for message in messages:
        src_role = message.get("role", "user")
        role = "model" if src_role == "assistant" else "user"
        content = message.get("content", "")
        parts: list[types.Part] = []

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError(f"Gemini content part must be dict: {part}")

                part_type = part.get("type")
                if part_type == "text":
                    parts.append(types.Part(text=str(part.get("text", ""))))
                    continue

                if part_type == "video":
                    video_path = part.get("video")
                    if not isinstance(video_path, str):
                        raise ValueError(f"Gemini video path must be string: {part}")
                    uploaded_file = upload_video(video_path)
                    if uploaded_file_names_out is not None:
                        fname = getattr(uploaded_file, "name", None)
                        if isinstance(fname, str) and fname:
                            uploaded_file_names_out.append(fname)
                    mime_type = getattr(uploaded_file, "mime_type", None) or "video/mp4"
                    file_uri = getattr(uploaded_file, "uri", None)
                    if file_uri is None:
                        raise ValueError(f"Gemini uploaded file uri missing: {video_path}")

                    part_fps = part.get("fps", default_fps)
                    if part_fps is None:
                        video_part = types.Part(
                            file_data=types.FileData(
                                file_uri=file_uri,
                                mime_type=mime_type,
                            )
                        )
                    else:
                        video_part = types.Part(
                            file_data=types.FileData(
                                file_uri=file_uri,
                                mime_type=mime_type,
                            ),
                            video_metadata=types.VideoMetadata(fps=part_fps),
                        )
                    parts.append(video_part)
                    continue

                raise ValueError(f"Unsupported content part format for Gemini: {part}")
        else:
            parts.append(types.Part(text=str(content)))

        inputs.append(types.Content(role=role, parts=parts))

    return inputs


# Legacy implementation (keep for reference):
# from google.genai import types
#
# def gemini_prepare_query(
#     query_video_path: str,
#     question_sample: str,
#     fps = None,
#     query_max_frames_num = None,
# ) -> dict[str, str]:
#     video_bytes = open(query_video_path, "rb").read()
#
#     # TODO Calculate fps based on the maximum frame count.
#     return [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part(
#                     inline_data=types.Blob(
#                         data=video_bytes,
#                         mime_type="video/mp4",
#                     ),
#                     video_metadata=types.VideoMetadata(fps=fps)
#                 ),
#                 types.Part(text=question_sample)
#             ]
#         )
#     ]
#
# def gemini_prepare_demonstration(
#     support_video_path: str,
#     question_sample: str,
#     ground_truth: list,
#     fps:int,
#     support_max_frames_num = None,
# ) -> dict[str, str]:
#     video_bytes = open(support_video_path, "rb").read()
#     ground_truth = str(ground_truth)
#
#     # TODO Calculate fps based on the maximum frame count.
#     return [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part(
#                     inline_data=types.Blob(
#                         data=video_bytes,
#                         mime_type="video/mp4",
#                     ),
#                     video_metadata=types.VideoMetadata(fps=fps)
#                 ),
#                 types.Part(text=question_sample)
#             ]
#         ),
#         types.Content(
#             role="model",
#             parts=[types.Part(text=ground_truth)]
#         ),
#     ]
