"""Convert raw spatio-temporal grounding prediction JSON files into Vidi evaluation.

This script performs the following steps:

1. Convert model output JSON files into `tubes.csv`.
2. Run Vidi evaluation with the matching GT `tubes.csv` / `tubes_fps1.csv`.
3. Save `summary.csv`.

For 2-shot runs, evaluation uses `data/{domain}/vidi_eval/` for the
corresponding domain instead of the individual dataset directory.
"""
import argparse
import ast
import csv
import json
import os.path as osp
import re
from datetime import datetime
from pathlib import Path

from decord import VideoReader, cpu

from src.eval.vidi_evaluate import (
    SpatioTemporalEvaluator,
    summarize_and_save_eval_simple,
)

DATASET_TO_DOMAIN = {
    "animal_kingdom": "animal",
    "mouse": "animal",
    "MECCANO": "industry",
    "meccano": "industry",
    "ENIGMA": "industry",
    "enigma": "industry",
    "american_football": "sports",
    "MultiSports": "sports",
    "multisports": "sports",
    "egosurgery": "surgery",
    "CholecTrack20": "surgery",
    "cholectrack20": "surgery",
    "uca": "safety",
    "DoTA": "safety",
    "dota": "safety",
}

DATASET_PATH_ALIASES = {
    "multisports": "MultiSports",
    "dota": "DoTA",
    "meccano": "MECCANO",
    "cholectrack20": "CholecTrack20",
    "enigma": "ENIGMA",
}

LLAVA_ST_BOX_PATTERN = re.compile(
    r"<TEMP-(\d{3})>:\[\s*"
    r"<WIDTH-(\d{3})><HEIGHT-(\d{3})><WIDTH-(\d{3})><HEIGHT-(\d{3})>"
    r"\s*\]"
)


def is_gpt_model(model: str | None) -> bool:
    """Return whether the model name belongs to the GPT family."""
    return bool(model) and model.startswith("gpt")


def is_gemini_model(model: str | None) -> bool:
    """Return whether the model name belongs to the Gemini family."""
    return bool(model) and model.startswith("gemini")


def is_llava_st_model(model: str | None) -> bool:
    """Return whether the model name belongs to the LLaVA-ST family."""
    return bool(model) and model.startswith("LLaVA-ST")


def parse_timestamp_to_seconds(value) -> float:
    """Convert numeric seconds or MM:SS / HH:MM:SS strings to seconds."""
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp")

    # Some model outputs use a timestamp range like "00:01-00:04"
    # even though downstream conversion expects a single anchor time.
    if "-" in text:
        text = text.split("-", 1)[0].strip()

    text = re.sub(r"\s*(seconds?|secs?)\s*$", "", text, flags=re.IGNORECASE)
    if text.lower().endswith("s"):
        text = text[:-1].strip()

    if ":" not in text:
        return float(text)

    parts = text.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp format: {value}")


def parse_datetime_timestamp(value) -> datetime | None:
    """Convert supported datetime strings to datetime objects or return None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_llava_st_boxes(text: str) -> list[dict]:
    """Parse LLaVA-ST <TEMP>/<WIDTH>/<HEIGHT> output into structured boxes."""
    rows = []
    for match in LLAVA_ST_BOX_PATTERN.finditer(text):
        frame_id = int(match.group(1))
        x0 = int(match.group(2)) / 99.0
        y0 = int(match.group(3)) / 99.0
        x1 = int(match.group(4)) / 99.0
        y1 = int(match.group(5)) / 99.0
        rows.append(
            {
                "frame_id": frame_id,
                "bbox_2d": [x0, y0, x1, y1],
            }
        )
    return rows


def parse_prediction_text(text: str) -> list[dict]:
    """Normalize model output text into bbox rows.

    Supported input formats:
    - Strings with JSON code fences.
    - Python dict/list-like strings.
    - Lists of `{"time": ..., "bbox_2d": [...]}`.
    - Dicts like `{"boxes": [{"timestamp": ..., "bbox": [...]}]}`.

    Returns a list of dicts where each item has `time` and `bbox_2d`.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    if text.endswith("```"):
        text = text[: -len("```")]
    text = text.strip()
    if not text or text == "None":
        return []
    llava_st_boxes = parse_llava_st_boxes(text)
    if llava_st_boxes:
        return llava_st_boxes
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, MemoryError):
        repaired_text = text
        repaired_text = re.sub(r"\[\s*\[([^\[\]]+)\]\s*\}", r"[\1]}", repaired_text)
        repaired_text = re.sub(r"\[\s*\[([^\[\]]+)\]\s*\]", r"[\1]", repaired_text)
        if repaired_text.count("{") > repaired_text.count("}"):
            repaired_text += "}" * (repaired_text.count("{") - repaired_text.count("}"))
        if repaired_text.count("[") > repaired_text.count("]"):
            repaired_text += "]" * (repaired_text.count("[") - repaired_text.count("]"))
        if repaired_text != text:
            try:
                parsed = json.loads(repaired_text)
                text = repaired_text
            except (json.JSONDecodeError, MemoryError):
                parsed = None
        else:
            parsed = None
    else:
        repaired_text = text
    if parsed is None:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError, MemoryError):
            repaired_text = re.sub(
                r'([\'"](?:timestamp|time|temporal_range)[\'"]\s*:\s*)(\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)',
                r'\1"\2"',
                text,
            )
            repaired_text = re.sub(
                r'(?<!["\'])\b\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b(?!["\'])',
                lambda m: f'"{m.group(0)}"',
                repaired_text,
            )
            if repaired_text != text:
                try:
                    parsed = ast.literal_eval(repaired_text)
                except (SyntaxError, ValueError, MemoryError):
                    parsed = None
                else:
                    text = repaired_text
            else:
                parsed = None
            if parsed is not None:
                pass
            else:
                pattern = re.compile(
                    r'\{"time":\s*([0-9]+(?:\.[0-9]+)?),\s*"bbox_2d":\s*\[([^\]]+)\]'
                )
                rows = []
                for time_str, bbox_str in pattern.findall(text):
                    try:
                        bbox = [float(part.strip()) for part in bbox_str.split(",")]
                    except ValueError:
                        continue
                    rows.append({"time": float(time_str), "bbox_2d": bbox})
                if rows:
                    return rows

                pattern = re.compile(
                    r'["\']?timestamp["\']?\s*:\s*["\']?'
                    r'(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?|\d+(?:\.\d+)?)["\']?'
                    r'.{0,80}?["\']?bbox_2d["\']?\s*:\s*\[\s*'
                    r'(-?\d+(?:\.\d+)?)\s*,\s*'
                    r'(-?\d+(?:\.\d+)?)\s*,\s*'
                    r'(-?\d+(?:\.\d+)?)\s*,\s*'
                    r'(-?\d+(?:\.\d+)?)',
                    re.DOTALL,
                )
                for match in pattern.finditer(text):
                    rows.append(
                        {
                            "time": parse_timestamp_to_seconds(match.group(1)),
                            "bbox_2d": [float(match.group(i)) for i in range(2, 6)],
                        }
                    )
                if rows:
                    return rows

                pattern = re.compile(
                    r'["\']?timestamp["\']?\s*:\s*["\']?'
                    r'(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?|\d+(?:\.\d+)?)["\']?'
                    r'.{0,120}?["\']?(?:bbox|box_2d|box)["\']?\s*:\s*\[+\s*'
                    r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
                    re.DOTALL,
                )
                for match in pattern.finditer(text):
                    rows.append(
                        {
                            "time": parse_timestamp_to_seconds(match.group(1)),
                            "bbox_2d": [float(match.group(i)) for i in range(2, 6)],
                        }
                    )
                if rows:
                    return rows

                pattern = re.compile(
                    r'(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?'
                    r'(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(?:\*\*)?\s*[:\-]\s*'
                    r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
                    re.DOTALL,
                )
                for match in pattern.finditer(text):
                    rows.append(
                        {
                            "time": parse_timestamp_to_seconds(match.group(1)),
                            "bbox_2d": [float(match.group(i)) for i in range(2, 6)],
                        }
                    )
                if rows:
                    return rows

                pattern = re.compile(
                    r'(?:at|by|where it is located at|situated at)\s*\[?'
                    r'(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\]?[^[]{0,120}'
                    r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',
                    re.IGNORECASE | re.DOTALL,
                )
                for match in pattern.finditer(text):
                    rows.append(
                        {
                            "time": parse_timestamp_to_seconds(match.group(1)),
                            "bbox_2d": [float(match.group(i)) for i in range(2, 6)],
                        }
                    )
                return rows

    if isinstance(parsed, dict) and "boxes" in parsed:
        rows = []
        for item in parsed["boxes"]:
            if not isinstance(item, dict):
                continue
            if "timestamp" in item and "bbox" in item:
                bbox_value = item["bbox"]
                if (
                    isinstance(bbox_value, list)
                    and len(bbox_value) == 1
                    and isinstance(bbox_value[0], list)
                ):
                    bbox_value = bbox_value[0]
                rows.append(
                    {
                        "time": parse_timestamp_to_seconds(item["timestamp"]),
                        "bbox_2d": bbox_value,
                    }
                )
            elif "timestamp" in item and "box" in item:
                bbox_value = item["box"]
                if (
                    isinstance(bbox_value, list)
                    and len(bbox_value) == 1
                    and isinstance(bbox_value[0], list)
                ):
                    bbox_value = bbox_value[0]
                rows.append(
                    {
                        "time": parse_timestamp_to_seconds(item["timestamp"]),
                        "bbox_2d": bbox_value,
                    }
                )
            elif "timestamp" in item and "box_2d" in item:
                bbox_value = item["box_2d"]
                if (
                    isinstance(bbox_value, list)
                    and len(bbox_value) == 1
                    and isinstance(bbox_value[0], list)
                ):
                    bbox_value = bbox_value[0]
                rows.append(
                    {
                        "time": parse_timestamp_to_seconds(item["timestamp"]),
                        "bbox_2d": bbox_value,
                    }
                )
            elif "timestamp" in item and "box2d" in item:
                bbox_value = item["box2d"]
                if (
                    isinstance(bbox_value, list)
                    and len(bbox_value) == 1
                    and isinstance(bbox_value[0], list)
                ):
                    bbox_value = bbox_value[0]
                rows.append(
                    {
                        "time": parse_timestamp_to_seconds(item["timestamp"]),
                        "bbox_2d": bbox_value,
                    }
                )
            elif "timestamp" in item and "box_2" in item:
                bbox_value = item["box_2"]
                if (
                    isinstance(bbox_value, list)
                    and len(bbox_value) == 1
                    and isinstance(bbox_value[0], list)
                ):
                    bbox_value = bbox_value[0]
                rows.append(
                    {
                        "time": parse_timestamp_to_seconds(item["timestamp"]),
                        "bbox_2d": bbox_value,
                    }
                )
            elif "timestamp" in item and "boxes" in item:
                bbox_value = item["boxes"]
                if (
                    isinstance(bbox_value, list)
                    and len(bbox_value) == 1
                    and isinstance(bbox_value[0], list)
                ):
                    bbox_value = bbox_value[0]
                rows.append(
                    {
                        "time": parse_timestamp_to_seconds(item["timestamp"]),
                        "bbox_2d": bbox_value,
                    }
                )
        return rows
    if isinstance(parsed, dict):
        if "timestamp" in parsed and "bbox_2d" in parsed:
            bbox_value = parsed["bbox_2d"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
        if "timestamp" in parsed and "box_2d" in parsed:
            bbox_value = parsed["box_2d"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
        if "timestamp" in parsed and "box2d" in parsed:
            bbox_value = parsed["box2d"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
        if "timestamp" in parsed and "bbox" in parsed:
            bbox_value = parsed["bbox"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
        if "timestamp" in parsed and "box" in parsed:
            bbox_value = parsed["box"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
        if "timestamp" in parsed and "box_2" in parsed:
            bbox_value = parsed["box_2"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
        if "timestamp" in parsed and "boxes" in parsed:
            bbox_value = parsed["boxes"]
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            return [
                {
                    "time": parse_timestamp_to_seconds(parsed["timestamp"]),
                    "bbox_2d": bbox_value,
                }
            ]
    if isinstance(parsed, list):
        if len(parsed) == 4 and all(isinstance(v, (int, float)) for v in parsed):
            return [{"time": 0.0, "bbox_2d": parsed}]
        return [item for item in parsed if isinstance(item, dict)]
    return []


def normalize_qwen_bbox(bbox: list[float]) -> list[float]:
    """Normalize Qwen-style 0-1000 bboxes into the 0-1 range."""
    raw_vals = [float(v) for v in bbox]
    if all(0.0 <= v <= 1.0 for v in raw_vals):
        vals = raw_vals
    else:
        vals = [v / 1000 for v in raw_vals]
    return [max(0.0, min(1.0, v)) for v in vals]


def normalize_gemini_bbox(bbox: list[float]) -> list[float]:
    """Normalize Gemini-style yxyx 0-1000 bboxes into xyxy 0-1 coordinates."""
    y0, x0, y1, x1 = [float(v) / 1000 for v in bbox]
    vals = [x0, y0, x1, y1]
    return [max(0.0, min(1.0, v)) for v in vals]


def normalize_gpt_bbox(bbox: list[float]) -> list[float]:
    """Clamp GPT-style 0-1 bboxes to the 0-1 range."""
    vals = [float(v) for v in bbox]
    return [max(0.0, min(1.0, v)) for v in vals]


def sample_video_frame_indices(video_path: str, fps: float, max_frames: int) -> tuple[list[int], float]:
    """Reproduce the video sampling rule used for GPT predictions."""
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = float(vr.get_avg_fps())
    sample_every = max(1, round(video_fps / fps))
    frame_indices = list(range(0, len(vr), sample_every))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    return frame_indices, video_fps


def get_video_duration_seconds(video_path: str) -> float:
    """Return the full video duration in seconds."""
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
    if video_fps <= 0:
        return 0.0
    return len(vr) / video_fps


def infer_dataset_from_prediction_path(source_json: Path) -> str:
    """Infer the dataset name from the suffix of a bundled prediction JSON path."""
    stem = source_json.stem
    for dataset in sorted(DATASET_TO_DOMAIN, key=len, reverse=True):
        if stem.endswith(f"_{dataset}"):
            return dataset
    raise ValueError(f"Cannot infer dataset from prediction path: {source_json}")


def resolve_query_video_path(source_json: Path, query_video_path: str) -> str:
    """Resolve the actual video path from a prediction sample query_video_path."""
    if osp.exists(query_video_path):
        return query_video_path
    dataset = infer_dataset_from_prediction_path(source_json)
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    video_name = query_video_path if query_video_path.endswith(".mp4") else f"{query_video_path}.mp4"
    return str(Path("data") / dataset_dir / "videos" / "clips" / video_name)


def resolve_query_video_path_for_task(
    source_json: Path,
    query_video_path: str,
    task: str,
) -> str:
    """Resolve the actual video path for evaluation according to the task."""
    if osp.exists(query_video_path):
        return query_video_path

    dataset = infer_dataset_from_prediction_path(source_json)
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    video_name = query_video_path if query_video_path.endswith(".mp4") else f"{query_video_path}.mp4"
    if task == "spatial":
        spatial_path = Path("data") / dataset_dir / "videos" / "clips4spatial" / video_name
        if spatial_path.exists():
            return str(spatial_path)
    return str(Path("data") / dataset_dir / "videos" / "clips" / video_name)


def build_prediction_text(sample: dict) -> str:
    """Extract the prediction text from one sample.

    Checks `prediction`, `pred`, and `raw_pred` in order, and uses the first
    element when the value is a list.
    """
    prediction = sample.get("prediction") or sample.get("pred")
    if isinstance(prediction, list) and prediction:
        return prediction[0]
    if isinstance(prediction, str):
        return prediction
    return sample.get("raw_pred") or ""


def _write_prediction_csv(target_csv: Path, rows: list[list[object]]) -> None:
    """Write one prediction CSV file."""
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with target_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "time_ms", "x0", "y0", "x1", "y1"])
        writer.writerows(rows)


def _write_parse_failures_csv(target_csv: Path, failed_rows: list[list[str]]) -> None:
    """Write parse failures to a CSV file for manual inspection."""
    failed_csv = target_csv.with_name(f"{target_csv.stem}_parse_failures.csv")
    with failed_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "raw_prediction"])
        writer.writerows(failed_rows)


def convert_prediction_json_to_vidi_tubes_default(
    source_json: Path,
    target_csv: Path,
    archive_csv: Path | None = None,
    model: str | None = None,
) -> None:
    """Convert Gemini/Qwen-style second plus 0-1000 bbox predictions to Vidi CSV."""
    data = json.loads(source_json.read_text())

    rows = []
    failed_rows = []
    for sample in data:
        query_id = sample["query_id"]
        prediction_text = build_prediction_text(sample)
        boxes = parse_prediction_text(prediction_text)
        if not boxes:
            rows.append([query_id, 0, 0.0, 0.0, 0.0, 0.0])
            failed_rows.append([query_id, prediction_text])
            continue

        base_datetime = None
        for item in boxes:
            if not isinstance(item, dict):
                continue
            candidate = item.get("time", item.get("timestamp"))
            base_datetime = parse_datetime_timestamp(candidate)
            if base_datetime is not None:
                break

        for item in boxes:
            if not isinstance(item, dict):
                continue
            if "time" in item and "bbox_2d" in item:
                time_value = item["time"]
                bbox_value = item["bbox_2d"]
            elif "timestamp" in item and "bbox_2d" in item:
                time_value = item["timestamp"]
                bbox_value = item["bbox_2d"]
            elif "timestamp" in item and "box" in item:
                time_value = item["timestamp"]
                bbox_value = item["box"]
            elif "timestamp" in item and "box_2d" in item:
                time_value = item["timestamp"]
                bbox_value = item["box_2d"]
            elif "timestamp" in item and "box2d" in item:
                time_value = item["timestamp"]
                bbox_value = item["box2d"]
            elif "timestamp" in item and "bbox" in item:
                time_value = item["timestamp"]
                bbox_value = item["bbox"]
            elif "timestamp" in item and "box_2" in item:
                time_value = item["timestamp"]
                bbox_value = item["box_2"]
            elif "timestamp" in item and "boxes" in item:
                time_value = item["timestamp"]
                bbox_value = item["boxes"]
            else:
                continue
            if (
                isinstance(bbox_value, list)
                and len(bbox_value) == 1
                and isinstance(bbox_value[0], list)
            ):
                bbox_value = bbox_value[0]
            if len(bbox_value) != 4:
                continue
            try:
                bbox_value = [float(v) for v in bbox_value]
            except (TypeError, ValueError):
                continue

            current_datetime = parse_datetime_timestamp(time_value)
            if base_datetime is not None and current_datetime is not None:
                time_seconds = (current_datetime - base_datetime).total_seconds()
            else:
                time_seconds = parse_timestamp_to_seconds(time_value)
            time_ms = int(round(time_seconds * 1000))
            if is_gemini_model(model):
                x0, y0, x1, y1 = normalize_gemini_bbox(bbox_value)
            else:
                x0, y0, x1, y1 = normalize_qwen_bbox(bbox_value)
            rows.append([query_id, time_ms, x0, y0, x1, y1])

    _write_prediction_csv(target_csv, rows)
    _write_parse_failures_csv(target_csv, failed_rows)
    if archive_csv is not None:
        _write_prediction_csv(archive_csv, rows)
        _write_parse_failures_csv(archive_csv, failed_rows)


def convert_prediction_json_to_vidi_tubes_gpt(
    source_json: Path,
    target_csv: Path,
    archive_csv: Path | None = None,
    task: str = "spatio_temporal",
) -> None:
    """Convert GPT-style frame_id plus 0-1 bbox predictions to timestamped Vidi CSV."""
    data = json.loads(source_json.read_text())

    rows = []
    failed_rows = []
    for sample in data:
        query_id = sample["query_id"]
        if task == "spatio_temporal":
            query_video_path = resolve_query_video_path(source_json, sample["query_video_path"])
            sampled_frame_indices, video_fps = sample_video_frame_indices(
                query_video_path,
                fps=1.0,
                max_frames=120,
            )
        prediction_text = build_prediction_text(sample)
        boxes = parse_prediction_text(prediction_text)
        if not boxes:
            rows.append([query_id, 0, 0.0, 0.0, 0.0, 0.0])
            failed_rows.append([query_id, prediction_text])
            continue

        for item in boxes:
            if not isinstance(item, dict):
                continue
            if "frame_id" in item and "bbox_2d" in item:
                time_value = item["frame_id"]
                bbox_value = item["bbox_2d"]
            elif "frame_id" in item and "bbox" in item:
                time_value = item["frame_id"]
                bbox_value = item["bbox"]
            elif "frame" in item and "box" in item:
                time_value = item["frame"]
                bbox_value = item["box"]
            elif "frame" in item and "bbox" in item:
                time_value = item["frame"]
                bbox_value = item["bbox"]
            elif "frame" in item and "bbox_2d" in item:
                time_value = item["frame"]
                bbox_value = item["bbox_2d"]
            elif "time" in item and "bbox_2d" in item:
                time_value = item["time"]
                bbox_value = item["bbox_2d"]
            elif "time" in item and "bbox" in item:
                time_value = item["time"]
                bbox_value = item["bbox"]
            else:
                continue
            if len(bbox_value) != 4:
                continue

            frame_id = int(round(float(time_value)))
            if task == "spatio_temporal":
                if frame_id < 0 or frame_id >= len(sampled_frame_indices):
                    continue
                source_frame_id = sampled_frame_indices[frame_id]
                time_ms = int(round(source_frame_id * 1000 / video_fps))
            else:
                time_ms = frame_id * 1000
            x0, y0, x1, y1 = normalize_gpt_bbox(bbox_value)
            rows.append([query_id, time_ms, x0, y0, x1, y1])

    _write_prediction_csv(target_csv, rows)
    _write_parse_failures_csv(target_csv, failed_rows)
    if archive_csv is not None:
        _write_prediction_csv(archive_csv, rows)
        _write_parse_failures_csv(archive_csv, failed_rows)


def convert_prediction_json_to_vidi_tubes_llava_st(
    source_json: Path,
    target_csv: Path,
    archive_csv: Path | None = None,
    task: str = "spatio_temporal",
) -> None:
    """Convert LLaVA-ST 100-bin spatio-temporal output to timestamped Vidi CSV."""
    data = json.loads(source_json.read_text())

    rows = []
    failed_rows = []
    for sample in data:
        query_id = sample["query_id"]
        query_video_path = resolve_query_video_path_for_task(
            source_json,
            sample["query_video_path"],
            task=task,
        )
        duration_sec = get_video_duration_seconds(query_video_path)
        prediction_text = build_prediction_text(sample)
        boxes = parse_prediction_text(prediction_text)
        if not boxes or duration_sec <= 0:
            rows.append([query_id, 0, 0.0, 0.0, 0.0, 0.0])
            failed_rows.append([query_id, prediction_text])
            continue

        for item in boxes:
            if not isinstance(item, dict):
                continue
            time_value = item.get("frame_id", item.get("frame", item.get("time")))
            bbox_value = item.get("bbox_2d", item.get("bbox", item.get("box")))
            if time_value is None or bbox_value is None or len(bbox_value) != 4:
                continue

            frame_id = min(max(int(round(float(time_value))), 0), 99)
            time_ms = int(round((duration_sec * frame_id / 99.0) * 1000))
            x0, y0, x1, y1 = normalize_gpt_bbox(bbox_value)
            rows.append([query_id, time_ms, x0, y0, x1, y1])

    _write_prediction_csv(target_csv, rows)
    _write_parse_failures_csv(target_csv, failed_rows)
    if archive_csv is not None:
        _write_prediction_csv(archive_csv, rows)
        _write_parse_failures_csv(archive_csv, failed_rows)


def convert_prediction_json_to_vidi_tubes(
    source_json: Path,
    target_csv: Path,
    archive_csv: Path | None = None,
    model: str | None = None,
    task: str = "spatio_temporal",
) -> None:
    """Select the appropriate prediction CSV conversion for each model."""
    if is_gpt_model(model):
        convert_prediction_json_to_vidi_tubes_gpt(
            source_json,
            target_csv,
            archive_csv,
            task=task,
        )
        return
    if is_llava_st_model(model):
        convert_prediction_json_to_vidi_tubes_llava_st(
            source_json,
            target_csv,
            archive_csv,
            task=task,
        )
        return
    convert_prediction_json_to_vidi_tubes_default(
        source_json,
        target_csv,
        archive_csv,
        model=model,
    )


def run_vidi_evaluation(
    dataset: str,
    model: str,
    step_ms: int = 1000,
    ignore_missing_pred: bool = False,
    task: str = "spatio_temporal",
) -> str:
    """Run Vidi evaluation with GT and prediction CSV files and save summary.csv.

    If both `tubes.csv` and `tubes_fps1.csv` exist on the GT side, both are
    evaluated and combined into the same summary with a `variant` column.
    """
    eval_root = osp.join("data", dataset, "vidi_eval")
    gt_dir = osp.join(eval_root, "vue-stg-benchmark")
    results_dir = osp.join(eval_root, "results")
    output_dir = osp.join(eval_root, "output")
    model_task = f"{model}_{task}"
    result_path = osp.join(results_dir, model_task, "tubes.csv")
    out_dir = osp.join(output_dir, model_task)

    evaluator = SpatioTemporalEvaluator(step_ms=step_ms)
    summaries = []
    if task == "spatial":
        variants = [("spatial_fps1", "tubes_spatial_fps1.csv")]
    else:
        variants = [("full_fps", "tubes.csv"), ("fps1", "tubes_fps1.csv")]

    for variant, tubes_file in variants:
        gt_tubes_path = osp.join(gt_dir, tubes_file)
        if not osp.exists(gt_tubes_path):
            continue
        print(f"GT file ({variant}): {gt_tubes_path}")
        evaluator.load_dataset(gt_dir, tubes_file=tubes_file)
        df = evaluator.evaluate_pred_file(
            result_path,
            ignore_missing_pred=ignore_missing_pred,
        )
        summaries.append((variant, df))

    return summarize_and_save_eval_simple(summaries, out_dir)


def parse_args() -> argparse.Namespace:
    """Define and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert raw STG predictions and run Vidi evaluation.")
    parser.add_argument("--task", choices=["spatio_temporal", "spatial"], default="spatio_temporal")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--step-ms", type=int, default=1000)
    parser.add_argument("--ignore-missing-pred", action="store_true")
    return parser.parse_args()


def resolve_eval_target_name(dataset: str, source_json: Path) -> str:
    """Resolve the dataset or domain name used as the evaluation target.

    Standard runs such as 0-shot use the dataset name unchanged. 2-shot runs
    have a source JSON stem ending in `_2shot`, so the dataset name is mapped
    to the corresponding domain name.
    """
    if source_json.stem.endswith("_2shot"):
        return DATASET_TO_DOMAIN.get(dataset, dataset)
    return dataset


def main() -> None:
    """Run conversion and evaluation end to end."""
    args = parse_args()
    eval_target_name = resolve_eval_target_name(args.dataset, args.source_json)
    target_csv = Path("data") / eval_target_name / "vidi_eval" / "results" / f"{args.model}_{args.task}" / "tubes.csv"
    print(f"Source JSON: {args.source_json}")
    print(f"Eval target: {eval_target_name}")
    print(f"Target CSV: {target_csv}")
    convert_prediction_json_to_vidi_tubes(
        args.source_json,
        target_csv,
        model=args.model,
        task=args.task,
    )
    summary_csv = run_vidi_evaluation(
        dataset=eval_target_name,
        model=args.model,
        step_ms=args.step_ms,
        ignore_missing_pred=args.ignore_missing_pred,
        task=args.task,
    )
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
