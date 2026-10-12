"""Spatio-temporal grounding AnyGroundBench"""
import json
import os

from tqdm import tqdm

from src.anygroundbench.base import BaseVideoICL
from src.anygroundbench.temporal import TemporalVideoICL
from src.message_builders import (
    build_demonstration_message_builder,
    build_query_message_builder,
)
from src.prompts.registry import build_prompt
from src.timestamp.not_interleave_text import add_duration, add_timestamp
from src.utils.setup_logger import set_logger

logger = set_logger()


class SpatioTemporalVideoICL(TemporalVideoICL):
    """Spatio-temporal grounding AnyGroundBench."""

    def __init__(self, args):
        """Initialize spatio-temporal grounding data and message builders."""
        BaseVideoICL.__init__(self, args)
        self.domain = args.domain_name
        self.annotation_dir = os.path.join("data", self.domain, "simrank")
        self.meta_data_dir = os.path.join("data", self.domain, "meta-data")
        self.video_data_dir = os.path.join("data", self.domain, "videos")

        if self.model_name == "gpt":
            spatio_temporal_label_key = "fps1_frame_id_0_1_demonstration"
            spatio_temporal_key_parser = int
            spatio_temporal_key_name = "frame_id"
            spatio_temporal_bbox_name = "box"
        elif self.model_name == "gemini":
            spatio_temporal_label_key = "fps1_timestamp_0_1000_yxyx_demonstration"
            spatio_temporal_key_parser = float
            spatio_temporal_key_name = "timestamp"
            spatio_temporal_bbox_name = "box_2d"
        elif self.model_name in ["internvl", "qwen3", "qwen3_5", "eagle"]:
            spatio_temporal_label_key = "fps1_timestamp_0_1000_st_demonstration"
            spatio_temporal_key_parser = float
            spatio_temporal_key_name = "timestamp"
            spatio_temporal_bbox_name = "bbox_2d"
        else:
            raise ValueError
        self.spatio_temporal_use_mmss = self.model_name == "gemini"
        self.spatio_temporal_key_name = spatio_temporal_key_name
        self.spatio_temporal_bbox_name = spatio_temporal_bbox_name

        self.train_video_to_entries = {}
        with open(os.path.join(self.meta_data_dir, "st_train.json"), encoding="utf-8") as f:
            train_data = json.load(f)
        for sample_id, entry in train_data.items():
            spatio_temporal_label = entry[spatio_temporal_label_key]
            if not spatio_temporal_label["tubes"]:
                continue
            start_str, end_str = entry["temporal_range"].split()
            self.train_video_to_entries[sample_id] = {
                "video_name": f'{entry["video_name"]}.mp4',
                "gt_time_range": [float(start_str), float(end_str)],
                "text": entry["text"].strip(),
                "spatio_temporal_label": spatio_temporal_label,
                "sampled_boxes": self._sample_boxes(
                    spatio_temporal_label,
                    spatio_temporal_key_parser,
                    spatio_temporal_key_name,
                ),
            }

        self.test_video_to_entries = {}
        with open(os.path.join(self.meta_data_dir, "st_test.json"), encoding="utf-8") as f:
            test_data = json.load(f)
        for sample_id, entry in test_data.items():
            spatio_temporal_label = entry[spatio_temporal_label_key]
            if not spatio_temporal_label["tubes"]:
                continue
            start_str, end_str = entry["temporal_range"].split()
            self.test_video_to_entries[sample_id] = {
                "video_name": f'{entry["video_name"]}.mp4',
                "gt_time_range": [float(start_str), float(end_str)],
                "text": entry["text"].strip(),
                "spatio_temporal_label": spatio_temporal_label,
                "sampled_boxes": self._sample_boxes(
                    spatio_temporal_label,
                    spatio_temporal_key_parser,
                    spatio_temporal_key_name,
                ),
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

    def prepare_messages(
        self,
        demonstrations,
        query_id,
        query_entries: dict | None = None,
    ):
        """Prepare support and query messages for spatio-temporal grounding."""
        demonstration_messages = []
        for demonstration in demonstrations:
            demonstration_info = self.train_video_to_entries[demonstration]
            video_name = demonstration_info["video_name"]
            demonstration_video_path = os.path.join(self.video_data_dir, "clips", video_name)
            demonstration_boxes = demonstration_info["sampled_boxes"]
            if self.spatio_temporal_use_mmss:
                demonstration_boxes = [
                    item for item in demonstration_boxes
                    if float(item["timestamp"]).is_integer()
                ]
            ground_truth = {
                "temporal_range": demonstration_info["gt_time_range"],
                "boxes": self._build_mllm_sampled_boxes(
                    demonstration_boxes,
                    self.spatio_temporal_key_name,
                    self.spatio_temporal_bbox_name,
                    self.spatio_temporal_use_mmss,
                ),
            }
            demonstration_prompt = build_prompt(
                task=self.task,
                model_name=self.model_name,
                domain_name=self.domain,
                prompt_variant=self.prompt_variant,
                query_text=demonstration_info["text"],
                # text_only=self.text_only_demonstration,
            )
            if "timestamp" in self.add_prompt:
                timestamp_prompt = add_timestamp(
                    video_path=demonstration_video_path,
                    fps=self.fps,
                    max_frames=self.support_max_frames_num,
                    model_name=self.model_name,
                )
                demonstration_prompt = timestamp_prompt + demonstration_prompt
            if "duration" in self.add_prompt:
                duration_prompt = add_duration(
                    video_path=demonstration_video_path,
                    model_name=self.model_name,
                )
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
        query_video_path = os.path.join(
            self.video_data_dir, "clips", query_info["video_name"]
        )
        query_prompt = build_prompt(
            task=self.task,
            model_name=self.model_name,
            domain_name=self.domain,
            prompt_variant=self.prompt_variant,
            query_text=query_info["text"],
        )
        if "timestamp" in self.add_prompt:
            timestamp_prompt = add_timestamp(
                video_path=query_video_path,
                fps=self.fps,
                max_frames=self.query_max_frames_num,
                model_name=self.model_name,
            )
            query_prompt = timestamp_prompt + query_prompt
        if "duration" in self.add_prompt:
            duration_prompt = add_duration(
                video_path=query_video_path,
                model_name=self.model_name,
            )
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
        """Run the spatio-temporal grounding inference pipeline."""
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
                prediction, _= self.process_inference(messages)

                query_entry = self.test_video_to_entries[query_id]
                ground_truth = {
                    "temporal_range": query_entry["gt_time_range"],
                    "boxes": self._build_mllm_sampled_boxes(
                        query_entry["sampled_boxes"],
                        self.spatio_temporal_key_name,
                        self.spatio_temporal_bbox_name,
                        self.spatio_temporal_use_mmss,
                    ),
                }
                prediction_records.append({
                    "query_id": query_id,
                    "query": query_entry["video_name"],
                    "demonstrations": demonstrations,
                    "gt": ground_truth,
                    "raw_pred": prediction,
                    "domain_name": self.domain,
                    "model_id": self.model_id,
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

        os.makedirs("results/spatio_temporal", exist_ok=True)
        result_path = os.path.join("results", "spatio_temporal", f"{self.now}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(prediction_records, f, indent=2, ensure_ascii=False)
        if errors:
            error_path = os.path.join("results", "spatio_temporal", f"{self.now}_errors.json")
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
        self.upload_run_record(result_path, valid_count=valid_count)
        return result_path

    def get_data(self):
        """Load similarity ranks and filtered spatio-temporal test query IDs."""
        similarity_rank, test_videos = super().get_data()
        test_videos = [sample_id for sample_id in test_videos if sample_id in self.test_video_to_entries]
        if self.target_query_ids:
            test_videos = [sample_id for sample_id in test_videos if sample_id in self.target_query_ids]
        if self.max_test_samples is not None:
            test_videos = test_videos[:self.max_test_samples]
        return similarity_rank, test_videos

    @classmethod
    def _format_mmss(cls, timestamp):
        """Format a timestamp as MM:SS."""
        total_seconds = max(0, int(round(float(timestamp))))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    @classmethod
    def _build_mllm_sampled_boxes(cls, boxes, key_name="frame_id", bbox_name="box", use_mmss=False):
        """Build MLLM-formatted sampled boxes."""
        mllm_boxes = []
        for item in boxes:
            time_value = item[key_name]
            if use_mmss and key_name == "timestamp":
                time_value = cls._format_mmss(time_value)
            mllm_boxes.append({
                key_name: time_value,
                bbox_name: item["bbox"],
            })
        return mllm_boxes

    @staticmethod
    def _sample_boxes(spatio_temporal_label, key_parser=int, key_name="frame_id"):
        """Extract sampled boxes from a spatio-temporal label."""
        sampled_boxes = []
        for tube in spatio_temporal_label["tubes"]:
            bbox_by_frame = tube["bbox"]
            frame_ids = sorted(bbox_by_frame, key=lambda x: key_parser(x))
            assert frame_ids, "bbox is empty"
            for frame_id in frame_ids:
                sampled_boxes.append({
                    key_name: key_parser(frame_id),
                    "bbox": bbox_by_frame[frame_id],
                })
        return sampled_boxes
