"""HTTP tool server for the agent: SAM3 (``tools.SAM3``) and Qwen3-VL-Embedding (``tools.Temporal.relevance``).

SAM3 part mirrors SpatialClaw's ``spatial_agent/gpu_models/sam3_model.py`` (SAM3Model):
  - SAM 3.0 ``Sam3VideoPredictor`` loaded from ``sam3.pt``; bf16 autocast entered per call.
  - ``/segment_image`` = ``SAM3Model._run_detect``: single image as a one-frame session;
    text -> detector path; points -> small boxes around each foreground point (detector
    path); boxes -> ``bounding_boxes`` (normalized xywh); ``output_prob_thresh=0.05``.
  - ``/segment_video`` = ``SAM3Model._run_segment_video``: ALL frames of
    [start_frame, end_frame) extracted to a JPEG dir; text prompts added one by one;
    points -> first foreground point per object (``obj_id`` = object index); boxes ->
    center point per object (box prompts do not register objects for propagation);
    propagate in both directions from ``prompt_frame_idx`` (local index).
  - Sessions are always closed in ``finally`` (otherwise session VRAM leaks).
  Instead of masks, responses carry the tight box around each mask (normalized
  [x0, y0, x1, y1]) plus the SAM3 score, since the agent's answer is a box per frame.

Embedding part (``/similarity``): Qwen3-VL-Embedding-8B text–video cosine similarity of a
video segment (frames sampled at ``fps``, at most ``max_frames``), as in TimeLens2's
semantic verification. Default instruction of the model for both sides.

GPU residency: the GPU is shared with the vLLM server, so the embedding model (16 GB)
lives on CPU and is moved to the GPU only for ``/similarity``; before any SAM3 call it is
moved back. With ``--swap_sam3`` SAM3 is swapped out the same way during embedding calls.
Requests are served one at a time, so GPU calls never overlap.
"""

import argparse
import collections
import base64
import gc
import glob
import io
import json
import os
import sys
import shutil
import tempfile
import time
import traceback
import select
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_CHECKPOINT = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt")
DEFAULT_EMBED_MODEL = os.path.join(
    os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")), "models--Qwen--Qwen3-VL-Embedding-8B/snapshots/*"
)


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    """Tight normalized [x0, y0, x1, y1] around a binary mask (None if empty)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


def parse_outputs(outputs: dict | None) -> list[dict]:
    """One frame's SAM3 outputs -> [{obj_id, box, score}] (boxes from masks)."""
    if outputs is None or len(outputs.get("out_obj_ids", [])) == 0:
        return []
    obj_ids = np.asarray(outputs["out_obj_ids"]).tolist()
    masks = outputs.get("out_binary_masks")
    probs = outputs.get("out_probs")
    records = []
    for j, obj_id in enumerate(obj_ids):
        box = mask_to_box(np.asarray(masks[j], dtype=bool))
        if box is None:
            continue
        score = float(probs[j]) if probs is not None and j < len(probs) else 1.0
        records.append({"obj_id": int(obj_id), "box": box, "score": score})
    return records


def video_info(video: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def extract_frames_to_dir(video: str, start_frame: int, end_frame: int, root: str, stride: int = 1) -> str:
    """Frames start_frame, start_frame + stride, ... (< end_frame) as JPEGs (SpatialClaw's _extract_frames_to_dir,
    which keeps every frame: stride 1)."""
    tmp_dir = tempfile.mkdtemp(prefix="sam3_frames_", dir=root)
    cap = cv2.VideoCapture(video)
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        saved = 0
        for offset in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if offset % stride:
                continue
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
                os.path.join(tmp_dir, f"{saved:06d}.jpg"), "JPEG", quality=95)
            saved += 1
    finally:
        cap.release()
    return tmp_dir


def read_frame(video: str, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        raise RuntimeError(f"cannot read frame {frame_index} of {video}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class Sam3Service:
    def __init__(self, checkpoint: str, tmp_root: str, max_video_frames: int, offload_video: bool = False):
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        print(f"[tools_server] loading SAM 3.0 checkpoint {checkpoint}", flush=True)
        self.predictor = Sam3VideoPredictor(checkpoint_path=checkpoint)
        self.tmp_root = tmp_root
        self.max_video_frames = max_video_frames
        self.offload_video = offload_video  # frames in CPU memory (slower); only needed when GPU memory is short

    def _request(self, req: dict) -> dict:
        return self.predictor.handle_request(req)

    def segment_image(self, video: str | None = None, frame_index: int | None = None, image_b64: str | None = None,
                      prompt: str | None = None, points=None, point_labels=None,
                      bounding_boxes=None, bounding_box_labels=None) -> dict:
        if image_b64 is not None:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        else:
            image = read_frame(video, int(frame_index))
        img_dir = tempfile.mkdtemp(prefix="sam3_img_", dir=self.tmp_root)
        outputs = None
        try:
            image.save(os.path.join(img_dir, "000000.jpg"), "JPEG", quality=95)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                session_id = self._request({"type": "start_session", "resource_path": img_dir})["session_id"]
                try:
                    req = {"type": "add_prompt", "session_id": session_id, "frame_index": 0,
                           "output_prob_thresh": 0.05}
                    if points is not None:
                        box_size = 0.05  # 5% of the image in each direction (as SpatialClaw)
                        boxes = []
                        for (cx, cy), lbl in zip(points, point_labels):
                            if lbl == 1:
                                xmin, ymin = max(0.0, cx - box_size), max(0.0, cy - box_size)
                                boxes.append([xmin, ymin, min(box_size * 2, 1.0 - xmin), min(box_size * 2, 1.0 - ymin)])
                        if boxes:
                            req["bounding_boxes"], req["bounding_box_labels"] = boxes, [1] * len(boxes)
                        else:
                            req["text"] = "object"
                    elif bounding_boxes is not None:
                        req["bounding_boxes"], req["bounding_box_labels"] = bounding_boxes, bounding_box_labels
                    else:
                        req["text"] = prompt
                    outputs = self._request(req)["outputs"]
                finally:
                    self._request({"type": "close_session", "session_id": session_id})
        finally:
            shutil.rmtree(img_dir, ignore_errors=True)
        return {"objects": parse_outputs(outputs)}

    def segment_video(self, video: str, start_frame: int | None = None, end_frame: int | None = None,
                      prompt_frame_idx: int = 0, prompts=None, points_per_object=None,
                      point_labels_per_object=None, boxes_per_object=None, frame_stride: int = 1) -> dict:
        """``frame_stride`` > 1 tracks on every stride-th frame only (prompt_frame_idx then counts those frames)."""
        fps, total = video_info(video)
        stride = max(1, int(frame_stride))
        start = 0 if start_frame is None else max(int(start_frame), 0)
        end = total if end_frame is None else min(int(end_frame), total)
        if end - start <= 0:
            raise ValueError(f"end_frame ({end}) must be greater than start_frame ({start}).")
        if end - start > self.max_video_frames:
            raise ValueError(f"Requested frame range ({start}-{end}) is {end - start} frames, which exceeds "
                             f"the maximum of {self.max_video_frames}. Use a narrower start_frame/end_frame window.")
        if not 0 <= prompt_frame_idx < -(-(end - start) // stride):
            raise ValueError(f"prompt_frame_idx={prompt_frame_idx} is outside the extracted range "
                             f"(0..{end - start - 1}, local to start_frame={start}).")
        geometric = points_per_object is not None or boxes_per_object is not None

        t_extract = time.time()
        frames_dir = extract_frames_to_dir(video, start, end, self.tmp_root, stride)
        t_extract = time.time() - t_extract
        t_infer = time.time()
        per_frame: dict[int, dict] = {}
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Decoded frames stay in CPU memory (moved to the GPU per frame): same outputs, no
                # per-frame GPU copy of the whole window, which ran the shared GPU out of memory.
                session_id = self._request({"type": "start_session", "resource_path": frames_dir,
                                            "offload_video_to_cpu": self.offload_video})["session_id"]
                try:
                    base = {"type": "add_prompt", "session_id": session_id, "frame_index": prompt_frame_idx}
                    if points_per_object is not None:
                        obj_idx = 0
                        for pts, lbls in zip(points_per_object, point_labels_per_object):
                            fg = [pt for pt, lbl in zip(pts, lbls) if lbl == 1]
                            if not fg:
                                continue
                            self._request({**base, "points": [fg[0]], "point_labels": [1], "obj_id": obj_idx})
                            obj_idx += 1
                    elif boxes_per_object is not None:
                        for obj_idx, (xmin, ymin, w, h) in enumerate(boxes_per_object):
                            self._request({**base, "points": [[xmin + w / 2, ymin + h / 2]],
                                           "point_labels": [1], "obj_id": obj_idx})
                    else:
                        for prompt in prompts:
                            self._request({**base, "text": prompt})
                    for item in self.predictor.handle_stream_request({
                        "type": "propagate_in_video", "session_id": session_id,
                        "propagation_direction": "both", "start_frame_index": prompt_frame_idx,
                        "max_frame_num_to_track": None,
                    }):
                        per_frame[item["frame_index"]] = item["outputs"]
                    if geometric and not per_frame:
                        raise RuntimeError("SAM3 video tracking returned no frames for the geometric prompt "
                                           "— object failed to register in the video session.")
                finally:
                    self._request({"type": "close_session", "session_id": session_id})
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

        objects: dict[int, dict] = {}
        for local_idx in sorted(per_frame):
            for rec in parse_outputs(per_frame[local_idx]):
                obj = objects.setdefault(rec["obj_id"], {"obj_id": rec["obj_id"], "boxes": {}, "scores": {}})
                obj["boxes"][start + local_idx * stride] = rec["box"]
                obj["scores"][start + local_idx * stride] = rec["score"]
        print(f"[tools_server] segment_video {end - start} frames (stride {stride}): extract {t_extract:.1f}s, "
              f"SAM3 {time.time() - t_infer:.1f}s", flush=True)
        return {"fps": fps, "frame_indices": list(range(start, end, stride)), "extract_sec": round(t_extract, 2),
                "sam3_sec": round(time.time() - t_infer, 2),
                "objects": [objects[k] for k in sorted(objects)]}


class EmbedService:
    """Qwen3-VL-Embedding kept on CPU; moved to the GPU only while computing a similarity."""

    def __init__(self, model_dir: str):
        sys.path.insert(0, model_dir)
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLForEmbedding
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        print(f"[tools_server] loading Qwen3-VL-Embedding from {model_dir} (on CPU)", flush=True)
        # Same attributes as Qwen3VLEmbedder.__init__, but the weights start on CPU.
        emb = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)
        emb.max_length, emb.fps, emb.default_instruction = 8192, 1, "Represent the user's input."
        import scripts.qwen3_vl_embedding as m
        emb.min_pixels, emb.max_pixels, emb.total_pixels = m.MIN_PIXELS, m.MAX_PIXELS, m.MAX_TOTAL_PIXELS
        emb.num_frames = emb.max_frames = m.MAX_FRAMES
        emb.model = Qwen3VLForEmbedding.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
        emb.processor = Qwen3VLProcessor.from_pretrained(model_dir, padding_side="right")
        self.embedder = emb
        self.on_gpu = False

    def to(self, device: str) -> None:
        want_gpu = device == "cuda"
        if want_gpu != self.on_gpu:
            if want_gpu:
                self.on_gpu = True  # set first: a move that OOMs half-way must still be undone by to("cpu")
                self.embedder.model.to(device)
            else:
                self.embedder.model.to(device)
                self.on_gpu = False
                gc.collect()
                torch.cuda.empty_cache()

    def similarity(self, video: str, text: str, start: float | None = None, end: float | None = None,
                   fps: float = 1.0, max_frames: int = 64, instruction: str | None = None,
                   boxes: dict | None = None, mode: str = "full", margin: float = 0.1,
                   dim: float = 0.3) -> dict:
        """Text–video cosine similarity of a segment, optionally focused on a tube.

        ``boxes`` ({frame_index: [x0, y0, x1, y1] normalized}) makes it spatio-temporal: the segment is the
        tube's frame span and each sampled frame uses the nearest tube box (within half a sampling step).
          mode="full"      whole frames (the temporal ``relevance``)
          mode="crop"      each frame cropped to its box enlarged by ``margin`` (A: object only)
          mode="highlight" whole frame, outside the box darkened to ``dim`` and the box outlined (B: context kept)
        Frames without a nearby box are dropped for crop and fully darkened for highlight.
        """
        vfps, total = video_info(video)
        if mode not in ("full", "crop", "highlight"):
            raise ValueError('mode must be "full", "crop" or "highlight"')
        track = {int(k): [float(v) for v in b] for k, b in (boxes or {}).items()}
        if mode != "full" and not track:
            raise ValueError(f'mode="{mode}" needs boxes')
        if track:
            s, e = min(track), max(track)
        else:
            s = max(0, int(round(float(start) * vfps)))
            e = min(total - 1, int(round(float(end) * vfps)))
        s, e = max(0, s), min(total - 1, e)
        if e < s:
            raise ValueError(f"empty segment [{start}, {end}]s")
        n = min(max_frames, max(2, int((e - s) / vfps * fps) + 1))
        indices = sorted(set(np.linspace(s, e, n).round().astype(int).tolist()))
        tol = max(1, int(round((e - s) / max(1, len(indices) - 1) / 2))) if track else 0
        keys = np.array(sorted(track)) if track else None
        frames = []
        for i in indices:
            img = read_frame(video, i)
            if mode == "full":
                frames.append(img)
                continue
            j = int(keys[np.argmin(np.abs(keys - i))])
            box = track[j] if abs(j - i) <= tol else None
            W, H = img.size
            if mode == "crop":
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                mx, my = (x1 - x0) * margin, (y1 - y0) * margin
                crop = [max(0, (x0 - mx) * W), max(0, (y0 - my) * H), min(W, (x1 + mx) * W), min(H, (y1 + my) * H)]
                if crop[2] - crop[0] < 8 or crop[3] - crop[1] < 8:
                    continue
                frames.append(img.crop(tuple(round(v) for v in crop)))
            else:
                arr = np.asarray(img).astype(np.float32) * dim
                if box is not None:
                    x0, y0, x1, y1 = (round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H))
                    arr[y0:y1, x0:x1] = np.asarray(img)[y0:y1, x0:x1]
                out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
                if box is not None:
                    from PIL import ImageDraw
                    ImageDraw.Draw(out).rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=max(2, H // 150))
                frames.append(out)
        if len(frames) < 2:
            raise ValueError(f"only {len(frames)} usable frame(s) for mode={mode}")
        if mode == "crop":  # the video processor needs one frame size
            w = max(f.width for f in frames); h = max(f.height for f in frames)
            frames = [f.resize((w, h)) for f in frames]
        self.embedder.num_frames = self.embedder.max_frames = len(frames)
        inputs = [{"text": text, "instruction": instruction}, {"video": frames, "instruction": instruction}]
        emb = self.embedder.process(inputs)
        sim = float((emb[0] * emb[1]).sum().item())
        return {"similarity": sim, "num_frames": len(frames), "frame_range": [s, e], "mode": mode}


class Residency:
    """Moves models between CPU and GPU so SAM3 and the embedding model never share the GPU peak."""

    def __init__(self, sam3: "Sam3Service", embed: EmbedService | None, swap_sam3: bool):
        self.sam3, self.embed, self.swap_sam3 = sam3, embed, swap_sam3
        self.sam3_on_gpu = True

    def use_sam3(self) -> None:
        if self.embed is not None:
            self.embed.to("cpu")
        if not self.sam3_on_gpu:
            self.sam3.predictor.model.to("cuda")
            self.sam3_on_gpu = True

    def use_embed(self) -> None:
        if self.swap_sam3 and self.sam3_on_gpu:
            self.sam3.predictor.model.to("cpu")
            self.sam3_on_gpu = False
            gc.collect()
            torch.cuda.empty_cache()
        self.embed.to("cuda")


GPU_LOCK = threading.Lock()

try:
    import ctypes
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # not glibc
    _LIBC = None


def _malloc_trim() -> None:
    """Return freed heap pages to the OS. Model swaps (CPU <-> GPU) and per-request threads leave glibc holding
    freed memory, so the server's RSS keeps growing."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def make_handler(service: Sam3Service, embed: EmbedService | None, residency: Residency):
    cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()

    def run_request(path: str, req: dict) -> dict:
        """Identical SAM3 requests (same video, window, prompts) return the cached result: SAM3 is
        deterministic here, and agents often re-run the same tracking (e.g. after a kernel reset)."""
        key = None
        if path in ("/segment_image", "/segment_video") and "image_b64" not in req:
            key = path + json.dumps(req, sort_keys=True)
            if key in cache:
                cache.move_to_end(key)
                return {**cache[key], "cached": True}
        out = _run_request(path, req)
        if key is not None:
            cache[key] = out
            while len(cache) > 256:
                cache.popitem(last=False)
        return out

    def _run_request(path: str, req: dict) -> dict:
        if path == "/segment_image":
            residency.use_sam3()
            return service.segment_image(**req)
        if path == "/segment_video":
            residency.use_sam3()
            return service.segment_video(**req)
        residency.use_embed()
        try:
            with torch.no_grad():
                return embed.similarity(**req)
        finally:
            # Back to CPU right away so SAM3 calls (and vLLM) keep their headroom between requests.
            embed.to("cpu")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"ok": True, "embedding": embed is not None,
                                  "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2)})
            else:
                self._reply(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                t0 = time.time()
                if self.path not in ("/segment_image", "/segment_video", "/similarity"):
                    return self._reply(404, {"error": f"unknown path {self.path}"})
                if self.path == "/similarity" and embed is None:
                    return self._reply(400, {"error": "embedding model not loaded (--no_embed)"})
                # Connections are accepted concurrently (so waiting clients do not hit connect timeouts);
                # the models run one request at a time.
                restore_offload = []
                with GPU_LOCK:
                    waited = round(time.time() - t0, 2)
                    if self._client_gone():  # the agent's cell timed out while this request was queued
                        print(f"[tools_server] skip {self.path}: client disconnected after {waited}s", flush=True)
                        return
                    try:
                        for attempt in range(2):  # one retry after freeing the cache on CUDA OOM
                            try:
                                out = run_request(self.path, req)
                                break
                            except torch.OutOfMemoryError:
                                if attempt:
                                    raise
                                traceback.print_exc()
                                gc.collect()
                                torch.cuda.empty_cache()
                                time.sleep(5)
                                if not service.offload_video:  # retry with the frames in CPU memory
                                    service.offload_video = True
                                    restore_offload.append(True)
                    finally:
                        if restore_offload:
                            service.offload_video = False
                        torch.cuda.empty_cache()
                        gc.collect()
                        _malloc_trim()  # give freed host memory back to the OS (RSS grew ~13 GB/h without it)
                out["elapsed_sec"] = round(time.time() - t0, 2)
                out["queued_sec"] = waited
                self._reply(200, out)
            except Exception as exc:  # report to the agent instead of crashing the server
                traceback.print_exc()
                try:
                    self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})
                except OSError:
                    pass

        def _client_gone(self) -> bool:
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def log_message(self, fmt, *args):
            print(f"[tools_server] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="sam3.pt path (glob allowed).")
    parser.add_argument("--tmp_root", default=None, help="Where frames are extracted (default: system temp).")
    parser.add_argument("--max_video_frames", type=int, default=1000, help="SpatialClaw's sam3_max_video_frames.")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Qwen3-VL-Embedding snapshot dir (glob allowed).")
    parser.add_argument("--no_embed", action="store_true", help="Serve SAM3 only.")
    parser.add_argument("--swap_sam3", action="store_true", help="Also move SAM3 to CPU during embedding calls.")
    parser.add_argument("--offload_video", action="store_true",
                        help="Keep SAM3 video frames in CPU memory (slower; for low GPU memory).")
    args = parser.parse_args()

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise FileNotFoundError(f"No SAM3 checkpoint at {args.checkpoint}")
    tmp_root = args.tmp_root or tempfile.gettempdir()
    os.makedirs(tmp_root, exist_ok=True)
    service = Sam3Service(matches[0], tmp_root, args.max_video_frames, offload_video=args.offload_video)
    embed = None
    if not args.no_embed:
        dirs = sorted(glob.glob(args.embed_model))
        if not dirs:
            raise FileNotFoundError(f"No embedding model at {args.embed_model}")
        embed = EmbedService(dirs[0])
    residency = Residency(service, embed, args.swap_sam3)
    ThreadingHTTPServer.request_queue_size = 64
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, embed, residency))
    print(f"[tools_server] ready on http://{args.host}:{args.port} (embedding={'on' if embed else 'off'}, "
          f"swap_sam3={args.swap_sam3})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
