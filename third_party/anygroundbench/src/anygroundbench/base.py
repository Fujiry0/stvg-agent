"""Provide the shared base class for AnyGroundBench inference."""

import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.models import load_model_and_processor
from src.utils.setup_logger import set_logger

logger = set_logger()

SPREADSHEET_ARG_KEYS = (
    "add_prompt",
    "alpha",
    "domain_name",
    "fps",
    "model_id",
    "model_name",
    "n_shot",
    "non_interleave_text",
    "overlay",
    "query_max_frames_num",
    "seed",
    "support_max_frames_num",
    "task",
    "text_only_demonstration",
)


class BaseVideoICL:
    """Base class for AnyGroundBench."""

    def __init__(self, args):
        """Initialize shared inference settings, model clients, and run metadata."""
        self.args = args
        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.domain_name = args.domain_name

        self.device = args.device
        self.n_shot = args.n_shot
        self.support_max_frames_num = args.support_max_frames_num
        self.query_max_frames_num = args.query_max_frames_num
        self.fps = args.fps
        self.task = args.task
        self.model_name = args.model_name
        self.model_id = args.model_id
        self.resize = (
            getattr(args, "resize", None)
            if self.model_name in ("gpt", "gemini")
            else None
        )
        if self.model_name == "gemini":
            from google.genai import (
                errors,  # Import here because each model uses a different environment.
            )
            self.model, self.gemini_model_name = load_model_and_processor(
                self.model_name, self.device, self.model_id
            )
            self.max_retries = args.max_retries
        elif self.model_name == "gpt":
            from src.models.gpt import (
                build_gpt_response_create_kwargs,
                extract_text_from_response,
            )
            self.model, self.gpt_model_name = load_model_and_processor(
                self.model_name, self.device, self.model_id
            )
            self.max_retries = args.max_retries
            self.build_gpt_response_create_kwargs = build_gpt_response_create_kwargs
            self.extract_gpt_text_from_response = extract_text_from_response
        else:
            self.model, self.processor = load_model_and_processor(
                self.model_name,
                self.device,
                self.model_id,
            )

        self.all_context_num = 0
        self.context_count = 0
        self.video_name_to_qa = None
        self.git_hash = self._get_git_hash()

    @staticmethod
    def _get_git_hash() -> str:
        """Return the current git commit hash if available."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    def upload_run_record(self, result_path: str, valid_count: int | None = None) -> None:
        """Upload run metadata to spreadsheet endpoint if configured."""
        try:
            if valid_count is not None and valid_count <= 0:
                logger.info("Skip run record upload because valid_count is 0.")
                return

            from dotenv import load_dotenv
            load_dotenv()
            endpoint = os.getenv("LB_URL")
            if not endpoint:
                return

            import requests

            payload = {
                "timestamp": self.now,
                "git_hash": self.git_hash,
                "result_path": result_path,
                "result_file": Path(result_path).name,
                "valid_count": valid_count,
                "args": {
                    key: getattr(self.args, key, None)
                    for key in SPREADSHEET_ARG_KEYS
                },
            }
            headers = {"Content-Type": "application/json"}

            response = requests.post(endpoint, json=payload, headers=headers)
            logger.info(f"Run record upload response: {response.status_code} {response.text}")
        except Exception as exc:
            logger.warning(f"Failed to upload run record: {exc}")

    def get_max_new_tokens(self) -> int:
        """Return the maximum number of generated tokens for the current task."""
        if self.task == "temporal":
            return 64
        if self.task in {"spatio_temporal", "spatial"}:
            return 4096
        raise ValueError("invalid task input")

    def prepare_inputs(self, messages: list[dict[str, Any]]):
        """Prepare model-specific inputs from chat messages."""
        if self.model_name == "qwen3":
            from src.models.qwen3 import get_qwen3vl_inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = get_qwen3vl_inputs(
                messages=messages,
                processor=self.processor,
                text=text,
            )
            inputs = inputs.to(self.device)
            token_num = inputs.input_ids.shape[1]

            return inputs, token_num

        elif self.model_name == "qwen3_5":
            from src.models.qwen3_5 import get_qwen3_5_inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True,
            )
            inputs = get_qwen3_5_inputs(
                messages=messages,
                processor=self.processor,
                text=text,
            )
            inputs = inputs.to(self.device)
            token_num = inputs.input_ids.shape[1]

            return inputs, token_num

        elif self.model_name == "internvl":
            from src.models.intern3 import get_intern3_inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = get_intern3_inputs(
                messages=messages,
                processor=self.processor,
                text=text,
            )
            inputs = inputs.to(self.device)
            token_num = inputs.input_ids.shape[1]

            return inputs, token_num

        elif self.model_name == "eagle":
            from src.models.eagle import get_eagle_inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = get_eagle_inputs(
                messages=messages,
                processor=self.processor,
                text=text,
            )
            inputs = inputs.to(self.device)
            token_num = inputs.input_ids.shape[1]

            return inputs, token_num

        elif self.model_name == "llava_st":
            from src.message_builders.llava_st_message import llava_st_to_conversations
            from src.models.llava_st import get_llava_st_inputs

            conversations, query_video_path, query_max_frames = llava_st_to_conversations(messages)
            query_max_frames = 100
            inputs = get_llava_st_inputs(
                conversations=conversations,
                query_video_path=query_video_path,
                query_max_frames=query_max_frames,
                llava_bundle=self.model,
            )
            token_num = int(inputs["input_ids"].shape[1])
            return inputs, token_num

        elif self.model_name == "gemini":
            from src.models.gemini import (
                cleanup_gemini_uploaded_files,
                get_gemini_inputs,
            )

            uploaded_file_names: list[str] = []
            try:
                inputs = get_gemini_inputs(
                    client=self.model,
                    messages=messages,
                    fps=self.fps,
                    resize=self.resize,
                    uploaded_file_names_out=uploaded_file_names,
                )
                token_response = self.model.models.count_tokens(
                    model=self.gemini_model_name,
                    contents=inputs,
                )
                token_num = token_response.total_tokens
            except Exception:
                cleanup_gemini_uploaded_files(self.model, uploaded_file_names)
                raise
            self._gemini_uploaded_file_names = uploaded_file_names
            return inputs, token_num

        elif self.model_name == "gpt":
            from src.models.gpt import get_gpt_inputs
            inputs = get_gpt_inputs(
                messages=messages,
                resize=self.resize,
            )
            # GPT input tokens are difficult to count exactly beforehand, so return 0.
            # Actual input tokens from responses.create() usage.input_tokens are added
            # to self.all_context_num in process_inference.
            return inputs, 0
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def process_inference(self, messages):
        """Run model inference for prepared messages."""
        inputs, token_num = self.prepare_inputs(messages)
        self.all_context_num += token_num
        self.context_count += 1

        if self.model_name in ["qwen3", "qwen3_5", "internvl"]:
           with torch.inference_mode():
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.get_max_new_tokens(),
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                generated_ids = generation_output.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                scores = torch.concat(list(generation_output.scores), dim=0)
                probs = F.softmax(scores, dim=-1)
                prob = probs.max(dim=-1).values.min().item()

        elif self.model_name == "eagle":
            with torch.inference_mode():
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                generated_ids = generation_output.sequences
                output_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                scores = torch.concat(list(generation_output.scores), dim=0)
                probs = F.softmax(scores, dim=-1)
                prob = probs.max(dim=-1).values.min().item()

        elif self.model_name == "llava_st":
            tokenizer = self.model["tokenizer"]
            llava_model = self.model["model"]
            with torch.inference_mode():
                output_ids = llava_model.generate(
                    inputs["input_ids"],
                    images=[inputs["image_tensors"]],
                    do_sample=True,
                    temperature=0.01,
                    top_p=None,
                    num_beams=1,
                    variables=inputs["variables"],
                    modalities=[inputs["modality"]],
                    max_new_tokens=self.get_max_new_tokens(),
                    use_cache=True,
                )
                output_text = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=False,
                )[0]
                output_text = output_text.replace("<|im_end|>", "").strip()
                prob = 0.9

        elif self.model_name == "gemini":
            from google.genai import errors as genai_errors

            from src.models.gemini import cleanup_gemini_uploaded_files

            uploaded_names = getattr(self, "_gemini_uploaded_file_names", [])
            try:
                for attempt in range(self.max_retries):
                    try:
                        response = self.model.models.generate_content(
                            model=self.gemini_model_name,
                            contents=inputs,
                        )
                        output_text = response.text
                        # return dummy probability
                        prob = 0.9
                        break

                    except genai_errors.APIError as e:
                        code = getattr(e, "code", None)
                        err_raw = str(e)
                        err_text = err_raw.lower()
                        # File storage exhaustion is not resolved by retrying.
                        storage_exhausted = (
                            code == 429
                            and "file_storage_bytes" in err_text
                        )
                        # Daily quota exhaustion is not resolved on the same day.
                        daily_quota_exhausted = (
                            code == 429
                            and (
                                "generate_requests_per_model_per_day" in err_text
                                or (
                                    "quota exceeded" in err_text
                                    and "per day" in err_text
                                )
                            )
                        )
                        if daily_quota_exhausted:
                            print(
                                "Gemini daily quota debug: "
                                f"code={code} model={self.gemini_model_name} "
                                f"max_retries={self.max_retries}"
                            )
                            print(f"Gemini daily quota raw error: {err_raw}")
                            logger.error(
                                "Gemini daily quota debug: code=%s model=%s max_retries=%s",
                                code,
                                self.gemini_model_name,
                                self.max_retries,
                            )
                            logger.error("Gemini daily quota raw error: %s", err_raw)
                            raise RuntimeError(
                                "Gemini daily quota exhausted "
                                f"(model={self.gemini_model_name}). "
                                "Skip retries and abort this run."
                            ) from e

                        is_retryable_status = (
                            code in {429, 500, 502, 503, 504}
                            and not storage_exhausted
                            and not daily_quota_exhausted
                        )
                        is_retryable_network = isinstance(
                            e.__cause__,
                            (TimeoutError, ConnectionError, OSError),
                        )

                        if not (is_retryable_status or is_retryable_network):
                            raise

                        if attempt == self.max_retries - 1:
                            raise

                        sleep_sec = min(2 ** attempt, 30) + random.uniform(0, 1)
                        time.sleep(sleep_sec)
            finally:
                cleanup_gemini_uploaded_files(self.model, uploaded_names)
                self._gemini_uploaded_file_names = []

        elif self.model_name == "gpt":
            import openai

            for attempt in range(self.max_retries):
                try:
                    response_kwargs = self.build_gpt_response_create_kwargs(
                        model_name=self.gpt_model_name,
                        inputs=inputs,
                        max_output_tokens=self.get_max_new_tokens(),
                    )
                    response = self.model.responses.create(**response_kwargs)
                    output_text = self.extract_gpt_text_from_response(response)
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        self.all_context_num += getattr(usage, "input_tokens", 0)
                    # return dummy probability
                    prob = 0.9
                    break

                except openai.APIConnectionError:
                    if attempt == self.max_retries - 1:
                        raise

                    sleep_sec = min(2 ** attempt, 30) + random.uniform(0, 1)
                    time.sleep(sleep_sec)

                except openai.RateLimitError:
                    if attempt == self.max_retries - 1:
                        raise

                    sleep_sec = min(2 ** attempt, 30) + random.uniform(0, 1)
                    time.sleep(sleep_sec)

                except openai.APIStatusError as e:
                    is_retryable_status = (
                        e.status_code in {408, 409} or e.status_code >= 500
                    )

                    if not is_retryable_status:
                        raise

                    if attempt == self.max_retries - 1:
                        raise

                    sleep_sec = min(2 ** attempt, 30) + random.uniform(0, 1)
                    time.sleep(sleep_sec)

        return output_text, prob
