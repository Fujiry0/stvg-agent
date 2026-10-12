"""Run object detection from annotations and first frames and output bbox JSON."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def arg_parse():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_paths", nargs="+", required=True)
    parser.add_argument("--first_frame_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--detector_backend",
        type=str,
        required=True,
        choices=["grounding_dino", "owlv2", "gemini"],
    )
    parser.add_argument("--detector_model_id", type=str, default=None)
    parser.add_argument("--grounding_dino_model_id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--owlv2_model_id", type=str, default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument("--gemini_max_retries", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def parse_annotation_line(line):
    """Parse one annotation line into video ID, timestamps, and query."""
    left, query = line.strip().split("##", 1)
    video_id, start_time, end_time = left.split()
    return {
        "video_id": video_id,
        "query": query,
        "start_time": float(start_time),
        "end_time": float(end_time),
    }
class GroundingDinoDetector:
    def __init__(self, model_id, device):
        """Initialize a Grounding DINO detector."""
        self.device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def detect(self, image, query, box_threshold, text_threshold):
        """Return Grounding DINO detections for an image and query."""
        prompt = query.strip()
        if not prompt.endswith("."):
            prompt = f"{prompt}."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "score": float(score),
                    "label": label,
                    "source": "grounding_dino",
                }
            )
        return detections


class OwlV2Detector:
    def __init__(self, model_id, device):
        """Initialize an OwlV2 detector."""
        self.device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def detect(self, image, query, box_threshold, _text_threshold):
        """Return OwlV2 detections for an image and query."""
        prompt = query.strip()
        if not prompt.endswith("."):
            prompt = f"{prompt}."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=[image.size[::-1]],
            threshold=box_threshold,
            text_labels=[[prompt]],
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "score": float(score),
                    "label": label,
                    "source": "owlv2",
                }
            )
        return detections



def build_detector(args):
    """Build a detector for the selected backend."""
    if args.detector_backend == "grounding_dino":
        model_id = args.detector_model_id or args.grounding_dino_model_id
        return GroundingDinoDetector(model_id=model_id, device=args.device)
    if args.detector_backend == "owlv2":
        model_id = args.detector_model_id or args.owlv2_model_id
        return OwlV2Detector(model_id=model_id, device=args.device)
    raise ValueError(f"Unsupported detector backend: {args.detector_backend}")


def main():
    """Run object detection on first frames and save bbox JSON files."""
    args = arg_parse()
    first_frame_root = Path(args.first_frame_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = build_detector(args)

    manifest = []
    for annotation_path_str in args.annotation_paths:
        annotation_path = Path(annotation_path_str)
        split_name = annotation_path.stem
        with open(annotation_path, encoding="utf-8") as f:
            entries = [parse_annotation_line(line) for line in f if line.strip()]
        if args.max_samples is not None:
            entries = entries[: args.max_samples]

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, entry in enumerate(tqdm(entries, desc=split_name)):
            output_path = split_dir / f"{sample_id:05d}.json"
            if args.skip_existing and output_path.exists():
                continue
            query_slug = "".join(
                char if char.isalnum() or char in {"_", "-"} else ""
                for char in entry["query"].lower().strip().replace(" ", "_")
            )[:80]
            image_path = first_frame_root / split_name / f"{sample_id:05d}_{entry['video_id']}_{query_slug}.png"
            image = Image.open(image_path).convert("RGB")
            detections = detector.detect(
                image=image,
                query=entry["query"],
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            record = {
                "sample_id": sample_id,
                "split": split_name,
                "video_id": entry["video_id"],
                "query": entry["query"],
                "timestamp": [entry["start_time"], entry["end_time"]],
                "image_path": str(image_path),
                "image_size": [image.width, image.height],
                "detector_backend": args.detector_backend,
                "detector_model_id": args.detector_model_id
                or (args.grounding_dino_model_id if args.detector_backend == "grounding_dino" else args.owlv2_model_id if args.detector_backend == "owlv2" else args.gemini_model),
                "detections": detections,
                "status": "ok" if detections else "no_detection",
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            manifest.append(record)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
