"""Generate SAM2 video tracks and visualizations from bbox JSON input."""
import argparse
import datetime
import json
import math
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torchvision.io import read_video, write_video
from torchvision.ops import masks_to_boxes
from tqdm import tqdm


def arg_parse():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_json_root", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--offload_video_to_cpu", action="store_true")
    parser.add_argument("--offload_state_to_cpu", action="store_true")
    parser.add_argument("--max_frame_num_to_track", type=int, default=None)
    parser.add_argument("--min_detection_score", type=float, default=0.0)
    parser.add_argument("--use_detection_index", type=int, default=0)
    parser.add_argument("--segment_input_mode", type=str, default="temp_mp4", choices=["temp_mp4", "memory"])
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--skip_segment_video_write", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def find_video_path(video_dir, video_id):
    """Find and return the mp4 path for a video ID."""
    direct_path = video_dir / f"{video_id}.mp4"
    if direct_path.exists():
        return direct_path
    matched_paths = sorted(video_dir.glob(f"*/{video_id}.mp4"))
    if matched_paths:
        return matched_paths[0]
    return direct_path


def load_video_segment(video_path, start_time, end_time):
    """Load video frames and FPS for a time segment."""
    video, _, info = read_video(
        str(video_path),
        start_pts=start_time,
        end_pts=end_time,
        pts_unit="sec",
        output_format="THWC",
    )
    fps = float(info["video_fps"])
    return video, fps


def annotate_frame(frame_tensor, frame_prediction):
    """Draw tracked boxes and scores on a frame."""
    image = Image.fromarray(frame_tensor.cpu().numpy().astype(np.uint8))
    draw = ImageDraw.Draw(image)
    object_ids = frame_prediction["object_ids"]
    boxes = frame_prediction["boxes"]
    scores = frame_prediction["scores"]
    for obj_id, box, score in zip(object_ids, boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0.0, y1 - 12.0)), f"id={int(obj_id)} score={float(score):.3f}", fill="red")
    return np.array(image, dtype=np.uint8)


def clamp_box(box, width, height):
    """Clamp a bbox to image bounds."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    return [x1, y1, x2, y2]


def make_segment_video(record, video_dir, segment_video_path):
    """Cut out a time segment and save it as a segment video."""
    video_path = find_video_path(video_dir, record["video_id"])
    segment_frames, fps = load_video_segment(video_path, record["timestamp"][0], record["timestamp"][1])
    if len(segment_frames) == 0:
        raise RuntimeError(f"empty decoded segment for {record['video_id']} sample={record['sample_id']}")
    segment_video_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(segment_video_path), segment_frames, fps=fps)
    return video_path, segment_frames, fps


def load_video_segment_only(record, video_dir):
    """Load only the video frames for a time segment."""
    video_path = find_video_path(video_dir, record["video_id"])
    segment_frames, fps = load_video_segment(video_path, record["timestamp"][0], record["timestamp"][1])
    if len(segment_frames) == 0:
        raise RuntimeError(f"empty decoded segment for {record['video_id']} sample={record['sample_id']}")
    return video_path, segment_frames, fps


def masks_to_frame_prediction(video_res_masks, object_ids, fallback_score):
    """Convert masks into per-frame bbox prediction format."""
    if isinstance(video_res_masks, np.ndarray):
        masks = torch.from_numpy(video_res_masks)
    else:
        masks = video_res_masks
    if masks.dim() == 4:
        masks = masks.squeeze(1)
    masks = masks > 0
    keep = masks.any(dim=(1, 2))
    keep_idx = torch.nonzero(keep, as_tuple=True)[0]
    if keep_idx.numel() == 0:
        return {
            "object_ids": torch.zeros(0, dtype=torch.int64),
            "scores": torch.zeros(0, dtype=torch.float32),
            "boxes": torch.zeros(0, 4, dtype=torch.float32),
            "masks": torch.zeros_like(masks[:0], dtype=torch.bool),
        }
    masks = torch.index_select(masks, 0, keep_idx)
    kept_ids = [object_ids[idx] for idx in keep_idx.tolist()]
    boxes = masks_to_boxes(masks)
    scores = torch.full((len(kept_ids),), float(fallback_score), dtype=torch.float32)
    return {
        "object_ids": torch.tensor(kept_ids, dtype=torch.int64),
        "scores": scores,
        "boxes": boxes,
        "masks": masks,
    }



def build_inference_state_from_frames(predictor, segment_frames, offload_video_to_cpu, offload_state_to_cpu):
    """Build a SAM2 inference state from in-memory frames."""
    compute_device = predictor.device
    video_height = int(segment_frames.shape[1])
    video_width = int(segment_frames.shape[2])

    images = segment_frames.permute(0, 3, 1, 2).float() / 255.0
    images = F.interpolate(
        images,
        size=(predictor.image_size, predictor.image_size),
        mode="bilinear",
        align_corners=False,
    )
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    images = (images - img_mean) / img_std

    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


def run_single_record(record, predictor, model_id, output_dir, video_dir, max_frame_num_to_track, min_detection_score, use_detection_index, offload_video_to_cpu, offload_state_to_cpu, segment_input_mode, skip_visualization, skip_segment_video_write):
    """Run SAM2 tracking for one bbox record."""
    detections = record.get("detections", [])
    if len(detections) <= use_detection_index:
        return {
            **record,
            "video_path": str(find_video_path(video_dir, record["video_id"])),
            "tracks": [],
            "status": "no_detection",
        }
    detection = detections[use_detection_index]
    if float(detection.get("score", 0.0)) < min_detection_score:
        return {
            **record,
            "video_path": str(find_video_path(video_dir, record["video_id"])),
            "tracks": [],
            "status": "below_score_threshold",
        }

    segment_video_dir = output_dir / "segment_videos" / record["split"] / record["video_id"]
    segment_video_path = segment_video_dir / f"{record['sample_id']:05d}.mp4"
    timings = {}

    segment_load_start = time.perf_counter()
    if segment_input_mode == "temp_mp4":
        video_path, segment_frames, fps = load_video_segment_only(record, video_dir)
        if not skip_segment_video_write:
            segment_video_path.parent.mkdir(parents=True, exist_ok=True)
            write_start = time.perf_counter()
            write_video(str(segment_video_path), segment_frames, fps=fps)
            timings["segment_write_sec"] = time.perf_counter() - write_start
    else:
        video_path, segment_frames, fps = load_video_segment_only(record, video_dir)
    timings["segment_load_sec"] = time.perf_counter() - segment_load_start

    init_start = time.perf_counter()
    if segment_input_mode == "temp_mp4":
        inference_state = predictor.init_state(
            video_path=str(segment_video_path),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
    else:
        inference_state = build_inference_state_from_frames(
            predictor=predictor,
            segment_frames=segment_frames,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
    timings["init_state_sec"] = time.perf_counter() - init_start

    init_box = clamp_box(
        detection["bbox_xyxy"],
        width=int(segment_frames.shape[2]),
        height=int(segment_frames.shape[1]),
    )
    add_prompt_start = time.perf_counter()
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        box=np.array(init_box, dtype=np.float32),
    )
    timings["add_prompt_sec"] = time.perf_counter() - add_prompt_start

    outputs_per_frame = {}
    propagate_start = time.perf_counter()
    for frame_idx, object_ids, video_res_masks in predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=max_frame_num_to_track,
    ):
        outputs_per_frame[int(frame_idx)] = masks_to_frame_prediction(
            video_res_masks=video_res_masks,
            object_ids=list(object_ids),
            fallback_score=float(detection.get("score", 1.0)),
        )
    timings["propagate_sec"] = time.perf_counter() - propagate_start

    start_frame = int(math.floor(record["timestamp"][0] * fps))
    end_frame = start_frame + len(segment_frames) - 1
    tracks = {}
    visualization_frames = []
    postprocess_start = time.perf_counter()
    for local_idx, frame_tensor in enumerate(segment_frames):
        frame_prediction = outputs_per_frame.get(local_idx)
        if frame_prediction is None:
            if not skip_visualization:
                visualization_frames.append(np.array(Image.fromarray(frame_tensor.cpu().numpy().astype(np.uint8)), dtype=np.uint8))
            continue
        object_ids = frame_prediction["object_ids"].tolist()
        boxes = frame_prediction["boxes"].tolist()
        scores = frame_prediction["scores"].tolist()
        absolute_frame_idx = start_frame + local_idx
        for tracked_obj_id, box, score in zip(object_ids, boxes, scores):
            if tracked_obj_id not in tracks:
                tracks[tracked_obj_id] = {"track_id": int(tracked_obj_id), "frames": []}
            tracks[tracked_obj_id]["frames"].append(
                {
                    "frame_idx": int(absolute_frame_idx),
                    "bbox_xyxy": [float(v) for v in box],
                    "score": float(score),
                }
            )
        if not skip_visualization:
            visualization_frames.append(annotate_frame(frame_tensor, frame_prediction))
    timings["postprocess_sec"] = time.perf_counter() - postprocess_start

    visualization_path = None
    if not skip_visualization:
        visualization_write_start = time.perf_counter()
        visualization_dir = output_dir / "visualizations" / record["split"]
        visualization_dir.mkdir(parents=True, exist_ok=True)
        visualization_path = visualization_dir / f"{record['sample_id']:05d}_{record['video_id']}.mp4"
        write_video(str(visualization_path), torch.from_numpy(np.stack(visualization_frames)), fps=fps)
        timings["visualization_write_sec"] = time.perf_counter() - visualization_write_start

    return {
        **record,
        "tracker_backend": "sam2_video_predictor",
        "model_id": model_id,
        "video_path": str(video_path),
        "segment_video_path": str(segment_video_path),
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "initial_bbox_xyxy": init_box,
        "timings": timings,
        "tracks": list(tracks.values()),
        "visualization_path": str(visualization_path) if visualization_path is not None else None,
        "status": "ok" if tracks else "no_track",
    }


def main():
    """Generate SAM2 tracking results for bbox JSON files in batch."""
    args = arg_parse()
    bbox_json_root = Path(args.bbox_json_root)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = predictor = SAM2VideoPredictor.from_pretrained(args.model_id, device=str(device))

    manifest = []
    error_records = []
    bbox_files = sorted(path for path in bbox_json_root.rglob("*.json") if path.name != "manifest.json")
    if args.max_samples is not None:
        bbox_files = bbox_files[: args.max_samples]

    for bbox_path in tqdm(bbox_files, desc="bbox_json"):
        with open(bbox_path, encoding="utf-8") as f:
            record = json.load(f)
        output_path = output_dir / "predictions" / record["split"] / record["video_id"] / f"{record['sample_id']:05d}.json"
        if args.skip_existing and output_path.exists():
            continue
        try:
            prediction = run_single_record(
                record=record,
                predictor=predictor,
                model_id=args.model_id,
                output_dir=output_dir,
                video_dir=video_dir,
                max_frame_num_to_track=args.max_frame_num_to_track,
                min_detection_score=args.min_detection_score,
                use_detection_index=args.use_detection_index,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
                segment_input_mode=args.segment_input_mode,
                skip_visualization=args.skip_visualization,
                skip_segment_video_write=args.skip_segment_video_write,
            )
            split_dir = output_dir / "predictions" / prediction["split"] / prediction["video_id"]
            split_dir.mkdir(parents=True, exist_ok=True)
            output_path = split_dir / f"{prediction['sample_id']:05d}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(prediction, f, ensure_ascii=False, indent=2)
            manifest.append(prediction)
        except Exception as exc:
            error_record = {
                **record,
                "bbox_json_path": str(bbox_path),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            error_records.append(error_record)
            print(
                f"[error] sample_id={record.get('sample_id')} video_id={record.get('video_id')} "
                f"error_type={type(exc).__name__}: {exc}"
            )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if error_records:
        error_output_path = output_dir / f"{run_timestamp}.json"
        with open(error_output_path, "w", encoding="utf-8") as f:
            json.dump(error_records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
