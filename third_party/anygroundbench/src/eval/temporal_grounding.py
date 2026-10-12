"""Evaluate temporal grounding prediction JSON files with IoU-based metrics."""

import argparse
import json
import re
from pathlib import Path


def iou(a, b):
    """Compute the IoU between two temporal intervals."""
    if a[0] == a[1] and b[0] == b[1]:
        return 1.0 if a[0] == b[0] else 0.0
    max0 = max((a[0]), (b[0]))
    min0 = min((a[0]), (b[0]))
    max1 = max((a[1]), (b[1]))
    min1 = min((a[1]), (b[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def extract_time(paragraph):
    """Extract temporal interval candidates from free-form text."""
    paragraph = paragraph.lower()

    timestamps = []

    # Check for HH:MM:SS and MM:SS formats FIRST (before checking individual numbers)
    # Also handle formats with optional milliseconds like MM:SS.xxx and HH:MM:SS.xx
    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        # convert to seconds
        time_matches_converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:  # HH:MM:SS.xx format
                h, m = map(int, parts[:2])
                s = float(parts[2])
                time_in_sec = h * 3600 + m * 60 + s
            elif len(parts) == 2:  # MM:SS.xxx format
                m = int(parts[0])
                s = float(parts[1])
                time_in_sec = m * 60 + s
            time_matches_converted.append(float(time_in_sec))
        timestamps = [
            (time_matches_converted[i], time_matches_converted[i + 1])
            for i in range(0, len(time_matches_converted), 2)
        ]

    # Check for The given query happens in m - n (seconds)
    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 18.5 - 23.0
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",  # 18.5 to 23.0
        ]
        for time_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [(float(start), float(end)) for start, end in time_matches]
                break

    # Check for other formats e.g.:
    # 1. Starting time: 0.8 seconds. Ending time: 1.1 seconds
    # 2. The start time for this event is 0 seconds, and the end time is 12 seconds.
    if len(timestamps) == 0:
        time_regex = re.compile(r"\b(\d+\.\d+|\d+)\b")  # time formats (e.g., 18, 18.5)
        time_matches = re.findall(time_regex, paragraph)
        time_matches = time_matches[: len(time_matches) // 2 * 2]
        timestamps = [
            (float(time_matches[i]), float(time_matches[i + 1]))
            for i in range(0, len(time_matches), 2)
        ]

    timestamps = [(start, end) for start, end in timestamps]
    return timestamps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="your_result.json")
    args = parser.parse_args()

    if args.f.endswith(".json"):
        with open(args.f) as fin:
            datas = json.load(fin)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .json or .jsonl file."
        )

    num_annos = len(datas)
    ious = []
    warning_counts = {
        "multiple_pairs": 0,
        "extraction_failed": 0,
        "invalid_timestamp": 0,
    }
    warning_lines = []

    # Example of annotation format:
    # anno = {
    #     f'{video_name}>>>{query}>>>{ground_truth_span}': {
    #         "timestamps": [(0, 10)],  # predicted time spans
    #         "answers": "The event happens from 0 - 10 seconds." # raw text answer from the model
    #     }
    # }
    for key, pred in datas.items():
        video_id, query, gt_span = key.split(">>>")
        gt_span = eval(gt_span)

        if type(pred) is dict:
            if "timestamps" in pred:
                timestamps = pred["timestamps"]
            elif "answers" in pred:  # parse the raw answer
                timestamps = extract_time(pred["answers"])
            else:
                raise ValueError(f"Unexpected key in prediction: {pred}")
        else:
            raise ValueError(
                f"Unexpected type for prediction: {type(pred)}. Expected dict or str."
            )

        if len(timestamps) > 1:
            warning_counts["multiple_pairs"] += 1
            warning_lines.append(
                f"multiple_pairs\tkey={key}\tpred={pred}\tused={timestamps[0]}"
            )
        elif len(timestamps) == 0:
            warning_counts["extraction_failed"] += 1
            warning_lines.append(
                f"extraction_failed\tkey={key}\tpred={pred}\ttimestamps={timestamps}"
            )
            timestamps = [(-100, -100)]

        timestamps = timestamps[0]  # only use the first pair of timestamps
        if timestamps[0] >= timestamps[1]:
            warning_counts["invalid_timestamp"] += 1
            warning_lines.append(
                f"invalid_timestamp\tkey={key}\tpred={pred}\ttimestamps={timestamps}"
            )

        ious.append(iou(gt_span, timestamps))

    recall = {0.3: 0, 0.5: 0, 0.7: 0}
    for iou_threshold in [0.3, 0.5, 0.7]:
        for cur_iou in ious:
            if cur_iou >= iou_threshold:
                recall[iou_threshold] += 1

    RESULT_STR = (
        f"IOU 0.3: {recall[0.3] * 100 / num_annos}\n"
        f"IOU 0.5: {recall[0.5] * 100 / num_annos}\n"
        f"IOU 0.7: {recall[0.7] * 100 / num_annos}\n"
        f"mIOU: {sum(ious) * 100 / num_annos}\n"
        f"mIOU: {sum(ious) * 100 / num_annos:.2f}"
    )

    print(RESULT_STR)
    print(
        "Warnings: "
        f"multiple_pairs={warning_counts['multiple_pairs']}, "
        f"extraction_failed={warning_counts['extraction_failed']}, "
        f"invalid_timestamp={warning_counts['invalid_timestamp']}"
    )

    # Save the result to a .log file
    log_file_path = Path(args.f).with_suffix(".log")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Processed file: {args.f}\n")
        log_file.write(RESULT_STR)
    warning_file_path = Path(args.f).with_suffix(".warnings.log")
    with open(warning_file_path, "w") as warning_file:
        warning_file.write(f"Processed file: {args.f}\n")
        warning_file.write(
            "Warning counts: "
            f"multiple_pairs={warning_counts['multiple_pairs']}, "
            f"extraction_failed={warning_counts['extraction_failed']}, "
            f"invalid_timestamp={warning_counts['invalid_timestamp']}\n"
        )
        for line in warning_lines:
            warning_file.write(line + "\n")
