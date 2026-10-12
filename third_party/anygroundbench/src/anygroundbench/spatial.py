"""Spatial grounding AnyGroundBench."""
import json
import os

from tqdm import tqdm

from src.anygroundbench.base import BaseVideoICL
from src.message_builders import (
    build_demonstration_message_builder,
    build_query_message_builder,
)
from src.prompts.registry import build_prompt, normalize_model_family
from src.prompts.spatial import build_spatial_prompt
from src.timestamp.not_interleave_text import add_duration, add_timestamp
from src.utils.setup_logger import set_logger

logger = set_logger()


class SpatialVideoICL(BaseVideoICL):
    """Spatial grounding AnyGroundBench."""

    def __init__(self, args):
        """Initialize spatial grounding data and message builders."""
        BaseVideoICL.__init__(self, args)
        self.domain = args.domain_name
        self.annotation_dir = os.path.join("data", self.domain, "simrank")
        self.meta_data_dir = os.path.join("data", self.domain, "meta-data")
        self.video_data_dir = os.path.join("data", self.domain, "videos")

        if self.model_name == "gpt":
            spatial_label_key = "fps1_frame_id_0_1_demonstration"
            spatial_key_parser = int
            spatial_key_name = "frame_id"
            spatial_box_key_name = "bbox"
        elif self.model_name == "gemini":
            spatial_label_key = "fps1_timestamp_0_1000_yxyx_demonstration"
            spatial_key_parser = float
            spatial_key_name = "timestamp"
            spatial_box_key_name = "box_2d"
        elif self.model_name in ["internvl", "qwen3", "qwen3_5", "eagle"]:
            spatial_label_key = "fps1_timestamp_0_1000_s_demonstration"
            spatial_key_parser = float
            spatial_key_name = "timestamp"
            spatial_box_key_name = "bbox_2d"
        else:
            raise ValueError
        self.spatial_key_name = spatial_key_name
        self.spatial_box_key_name = spatial_box_key_name

        self.train_video_to_entries = {}
        with open(os.path.join(self.meta_data_dir, "s_train.json"), encoding="utf-8") as f:
            train_data = json.load(f)
        for sample_id, entry in train_data.items():
            spatial_label = entry[spatial_label_key]
            sampled_boxes = self._sample_boxes(
                spatial_label,
                spatial_key_parser,
                spatial_key_name,
            )
            if not sampled_boxes:
                continue
            self.train_video_to_entries[sample_id] = {
                "video_name": f'{entry["video_name"]}.mp4',
                "clips4spatial_name": (
                    f'{entry["clips4spatial_name"]}.mp4' if entry.get("clips4spatial_name") else None
                ),
                "text": entry["text"].strip(),
                "spatial_label": spatial_label,
                "sampled_boxes": sampled_boxes,
            }

        self.test_video_to_entries = {}
        with open(os.path.join(self.meta_data_dir, "s_test.json"), encoding="utf-8") as f:
            test_data = json.load(f)
        for sample_id, entry in test_data.items():
            spatial_label = entry[spatial_label_key]
            sampled_boxes = self._sample_boxes(
                spatial_label,
                spatial_key_parser,
                spatial_key_name,
            )
            if not sampled_boxes:
                continue
            self.test_video_to_entries[sample_id] = {
                "video_name": f'{entry["video_name"]}.mp4',
                "clips4spatial_name": (
                    f'{entry["clips4spatial_name"]}.mp4' if entry.get("clips4spatial_name") else None
                ),
                "text": entry["text"].strip(),
                "spatial_label": spatial_label,
                "sampled_boxes": sampled_boxes,
            }

        self.alpha = args.alpha
        self.non_interleave_text = args.non_interleave_text
        self.overlay = args.overlay
        self.add_prompt = args.add_prompt
        self.prompt_variant = args.prompt_variant
        self.text_only_demonstration = args.text_only_demonstration
        self.text_only = False
        if self.text_only_demonstration:
            self.text_only = True
        self.max_test_samples = args.max_test_samples
        self.target_query_ids = set(args.target_query_ids or [])

        self.build_demonstration_message = build_demonstration_message_builder(
            self.model_name,
            text_only=self.text_only_demonstration,
        )
        self.build_query_message = build_query_message_builder(
            self.model_name,
            text_only=self.text_only
        )

    def get_data(self):
        """Load similarity ranks and spatial test query IDs."""
        if self.n_shot == 0:
            test_videos = list(self.test_video_to_entries.keys())
            if self.target_query_ids:
                test_videos = [sample_id for sample_id in test_videos if sample_id in self.target_query_ids]
            if self.max_test_samples is not None:
                test_videos = test_videos[:self.max_test_samples]
            return {}, test_videos

        simrank_path = os.path.join("data", self.domain, "simrank", f"video_top100_alpha{self.alpha}.json")
        with open(simrank_path) as f:
            annotation = json.load(f)

        similarity_rank = {}
        for row in annotation:
            similarity_rank[row["test_sample"]] = row["train_examples"]

        test_videos = [row["test_sample"] for row in annotation]
        test_videos = [sample_id for sample_id in test_videos if sample_id in self.test_video_to_entries]
        if self.target_query_ids:
            test_videos = [sample_id for sample_id in test_videos if sample_id in self.target_query_ids]
        if self.max_test_samples is not None:
            test_videos = test_videos[:self.max_test_samples]
        return similarity_rank, test_videos

    @staticmethod
    def _sample_boxes(spatial_label, key_parser=int, key_name="frame_id"):
        """Extract sampled boxes from a spatial label."""
        if "tubes" in spatial_label:
            boxes = []
            for tube in spatial_label["tubes"]:
                for frame_id, bbox in sorted(tube["bbox"].items(), key=lambda item: key_parser(item[0])):
                    boxes.append({
                        key_name: key_parser(frame_id),
                        "bbox": bbox,
                        "tube_index": int(tube.get("tube_index", 0)),
                    })
            return boxes
        return [{
            key_name: key_parser(spatial_label["frame_id"]),
            "bbox": spatial_label["bbox"],
        }]

    def _get_selected_frame_ids(self, video_path, max_frames_num):
        """Return selected frame IDs and source frame IDs for a video."""
        messages = self.build_query_message(
            query_video_path=video_path,
            query_max_frames_num=max_frames_num,
            fps=self.fps,
            question_sample="",
        )
        if self.model_name == "eagle":
            # Lazy import to isolate dependency environments.
            from src.models.eagle import process_eagle_vision_info
            vision_info = process_eagle_vision_info(
                processor=self.processor,
                messages=messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                fps=self.fps,
            )
        else:
            from qwen_vl_utils import process_vision_info

            vision_info = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        _, videos, _ = vision_info
        selected_video_inputs, video_metadata = videos[-1]
        frame_count = int(selected_video_inputs.shape[0])
        source_frame_ids = [int(frame_id) for frame_id in video_metadata["frames_indices"]]
        return list(range(frame_count)), source_frame_ids

    @classmethod
    def _format_mmss(cls, timestamp):
        """Format a timestamp as MM:SS."""
        total_seconds = max(0, int(round(float(timestamp))))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    @classmethod
    def _build_mllm_sampled_boxes(
        cls,
        entry,
        selected_source_frame_ids,
        key_name="frame_id",
        box_key_name="bbox",
    ):
        """Map sampled boxes to the frame or timestamp format expected by MLLMs."""
        boxes = entry["sampled_boxes"]
        mllm_boxes = []
        for item in boxes:
            frame_id = item.get("frame_id", item.get("timestamp"))
            if frame_id is None:
                raise KeyError("frame_id")
            frame_id = int(frame_id)
            nearest_idx = min(
                range(len(selected_source_frame_ids)),
                key=lambda idx: abs(selected_source_frame_ids[idx] - frame_id),
            )
            time_value = item.get(key_name, nearest_idx)
            if key_name == "timestamp" and box_key_name == "box_2d":
                time_value = cls._format_mmss(time_value)
            mllm_boxes.append({
                key_name: time_value,
                box_key_name: item["bbox"],
            })
        return mllm_boxes

    def prepare_messages(
        self,
        demonstrations,
        query_id,
        query_entries: dict | None = None,
    ):
        """Prepare support and query messages for spatial grounding."""
        demonstration_messages = []
        for demonstration in demonstrations:
            demonstration_info = self.train_video_to_entries[demonstration]
            video_name = demonstration_info["video_name"]
            demonstration_video_path = os.path.join(self.video_data_dir, "clips4spatial", video_name)
            if self.text_only:
                selected_source_frame_ids = [
                    int(item.get("frame_id", item.get("timestamp")))
                    for item in demonstration_info["sampled_boxes"]
                ]
            else:
                selected_frame_ids, selected_source_frame_ids = self._get_selected_frame_ids(
                    demonstration_video_path,
                    self.support_max_frames_num,
                )
            demonstration_info["spatial_mllm_sampled_boxes"] = self._build_mllm_sampled_boxes(
                demonstration_info,
                selected_source_frame_ids,
                self.spatial_key_name,
                self.spatial_box_key_name,
            )
            ground_truth = json.dumps({"boxes": demonstration_info["spatial_mllm_sampled_boxes"]}, ensure_ascii=False)
            if normalize_model_family(self.model_name) == "llava_st":
                demonstration_prompt = build_spatial_prompt(
                    model_family="llava_st",
                    query_text=demonstration_info["text"],
                ).text
            else:
                demonstration_prompt = build_prompt(
                    task=self.task,
                    model_name=self.model_name,
                    domain_name=self.domain,
                    prompt_variant=self.prompt_variant,
                    query_text=demonstration_info["text"],
                ).text
            if "timestamp" in self.add_prompt:
                timestamp_prompt = add_timestamp(
                    video_path=demonstration_video_path,
                    fps=self.fps,
                    max_frames=self.support_max_frames_num,
                    model_name=self.model_name,
                )
                demonstration_prompt = timestamp_prompt + demonstration_prompt
            if "duration" in self.add_prompt:
                duration_prompt = add_duration(video_path=demonstration_video_path)
                demonstration_prompt = duration_prompt + demonstration_prompt
            demonstration_messages += self.build_demonstration_message(
                support_video_path=demonstration_video_path,
                support_max_frames_num=self.support_max_frames_num,
                fps=self.fps,
                question_sample=demonstration_prompt,
                ground_truth=ground_truth,
            )

        query_entries = self.test_video_to_entries if query_entries is None else query_entries
        query_info = query_entries[query_id]
        video_name = query_info["video_name"]
        query_video_path = query_info.get(
            "video_path",
            os.path.join(self.video_data_dir, "clips4spatial", video_name),
        )
        if self.text_only:
            selected_source_frame_ids = [
                int(item.get("frame_id", item.get("timestamp")))
                for item in query_info["sampled_boxes"]
            ]
        else:
            selected_frame_ids, selected_source_frame_ids = self._get_selected_frame_ids(
                query_video_path,
                self.query_max_frames_num,
            )
        query_info["spatial_mllm_sampled_boxes"] = self._build_mllm_sampled_boxes(
            query_info,
            selected_source_frame_ids,
            self.spatial_key_name,
            self.spatial_box_key_name,
        )
        if normalize_model_family(self.model_name) == "llava_st":
            query_prompt = build_spatial_prompt(
                model_family="llava_st",
                query_text=query_info["text"],
            ).text
        else:
            query_prompt = build_prompt(
                task=self.task,
                model_name=self.model_name,
                domain_name=self.domain,
                prompt_variant=self.prompt_variant,
                query_text=query_info["text"],
            ).text
        if "timestamp" in self.add_prompt:
            timestamp_prompt = add_timestamp(
                video_path=query_video_path,
                fps=self.fps,
                max_frames=self.query_max_frames_num,
                model_name=self.model_name,
            )
            query_prompt = timestamp_prompt + query_prompt
        if "duration" in self.add_prompt:
            duration_prompt = add_duration(video_path=query_video_path)
            query_prompt = duration_prompt + query_prompt
        query_messages = self.build_query_message(
            query_video_path=query_video_path,
            query_max_frames_num=self.query_max_frames_num,
            fps=self.fps,
            question_sample=query_prompt,
        )
        messages = demonstration_messages + query_messages
        # print(messages)
        return messages

    def inference(self):
        """Run the spatial grounding inference pipeline."""
        sim_rank, test_videos = self.get_data()

        valid_count = 0
        prediction_records = []
        errors = []
        for query_id in tqdm(test_videos):
            # try:
                if self.n_shot == 0:
                    demonstrations = []
                else:
                    candidates = sim_rank[query_id.split("/")[-1]]
                    demonstrations = candidates[:self.n_shot]
                messages = self.prepare_messages(demonstrations, query_id)
                prediction, _ = self.process_inference(messages)

                query_entry = self.test_video_to_entries[query_id]
                ground_truth = json.dumps({"boxes": query_entry["spatial_mllm_sampled_boxes"]}, ensure_ascii=False)
                prediction_records.append({
                    "query_id": query_id,
                    "query": query_entry["video_name"],
                    "demonstrations": demonstrations,
                    "gt": ground_truth,
                    "raw_pred": prediction,
                    "domain_name": self.domain,
                    "model_name": self.model_name,
                    "n_shot": self.n_shot,
                    "text_only_demonstration": self.text_only_demonstration,
                    "query_video_path": query_entry["video_name"],
                })
                valid_count += 1
            # except Exception as e:
            #     query_entry = self.test_video_to_entries.get(query_id, {})
            #     short_error = f"{type(e).__name__}: {str(e)}"
            #     errors.append({
            #         "query_id": query_id,
            #         "video_name": query_entry.get("video_name"),
            #         "error": short_error,
            #     })

        avg_context_num = (
            round(self.all_context_num / self.context_count, 5)
            if self.context_count > 0
            else 0.0
        )
        logger.info(f"Avg Context Number: {avg_context_num}")
        logger.info(f"Valid Count: {valid_count}")

        os.makedirs("results/spatial", exist_ok=True)
        result_path = os.path.join("results", "spatial", f"{self.now}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(prediction_records, f, indent=2, ensure_ascii=False)
        if errors:
            error_path = os.path.join("results", "spatial", f"{self.now}_errors.json")
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
        self.upload_run_record(result_path, valid_count=valid_count)
        return result_path
