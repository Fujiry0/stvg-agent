import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.spatio_temporal_grounding import (
    convert_prediction_json_to_vidi_tubes,
    run_vidi_evaluation,
)
from src.eval.vidi_evaluate import SpatioTemporalEvaluator

DATASET_TO_DOMAIN = {
    "animal_kingdom": "animal",
    "mouse": "animal",
    "meccano": "industry",
    "enigma": "industry",
    "multisports": "sports",
    "american_football": "sports",
    "egosurgery": "surgery",
    "cholectrack20": "surgery",
    "uca": "safety",
    "dota": "safety",
}

DOMAIN_DATASET_ORDER = [
    ("animal", "animal_kingdom"),
    ("animal", "mouse"),
    ("industry", "meccano"),
    ("industry", "enigma"),
    ("sports", "multisports"),
    ("sports", "american_football"),
    ("surgery", "egosurgery"),
    ("surgery", "cholectrack20"),
    ("safety", "uca"),
    ("safety", "dota"),
]
DOMAINS = tuple(dict.fromkeys(domain for domain, _dataset in DOMAIN_DATASET_ORDER))

DATASET_PATH_ALIASES = {
    "multisports": "MultiSports",
    "dota": "DoTA",
    "meccano": "MECCANO",
    "cholectrack20": "CholecTrack20",
    "enigma": "ENIGMA",
}

META_KEYS = [
    "variant",
    "group",
    "category",
]

GT_PAYLOAD_KEYS = [
    "spatio_temporal_label",
    "timestamp_0_1000_st_demonstration",
    "fps1_timestamp_0_1000_st_demonstration",
    "fps1_timestamp_0_1000_yxyx_demonstration",
    "frame_id_0_1_demonstration",
    "fps1_frame_id_0_1_demonstration",
]

SPATIAL_GT_PAYLOAD_KEYS = [
    "spatial_label",
    "fps1_timestamp_0_1000_yxyx_demonstration",
    "fps1_frame_id_0_1_demonstration",
    "fps1_timestamp_0_1000_s_demonstration",
]


def write_yxyx_csv_from_xyxy_csv(target_csv: Path) -> None:
    """xyxy の GT tubes CSV から yxyx 列順の sibling CSV を作る。"""
    yxyx_csv = target_csv.with_name(f"{target_csv.stem}_yxyx.csv")
    with target_csv.open(newline="") as src_f, yxyx_csv.open("w", newline="") as dst_f:
        reader = csv.DictReader(src_f)
        writer = csv.writer(dst_f)
        writer.writerow(["query_id", "time_ms", "y0", "x0", "y1", "x1"])
        for row in reader:
            writer.writerow(
                [
                    row["query_id"],
                    row["time_ms"],
                    row["y0"],
                    row["x0"],
                    row["y1"],
                    row["x1"],
                ]
            )


def infer_prediction_target(path: Path) -> tuple[str, str]:
    """予測 JSON のファイル名末尾から dataset または domain 名を推定する。"""
    stem = path.stem
    shot_suffix = get_shot_suffix(stem)
    if shot_suffix is not None:
        stem = stem[: -len(shot_suffix)]
    for dataset in sorted(DATASET_TO_DOMAIN, key=len, reverse=True):
        if stem.endswith(f"_{dataset}"):
            return "dataset", dataset
    for domain in sorted(DOMAINS, key=len, reverse=True):
        if stem.endswith(f"_{domain}"):
            return "domain", domain
    raise ValueError(f"Cannot infer dataset or domain from prediction path: {path}")


def domain_for_target(target: str) -> str:
    """dataset/domain 名から domain 名を返す。"""
    return target if target in DOMAINS else DATASET_TO_DOMAIN[target]


def is_domain_target(target: str) -> bool:
    """評価対象名が domain かどうかを返す。"""
    return target in DOMAINS


def get_shot_suffix(stem: str) -> str | None:
    """ファイル名末尾の shot suffix を返す。"""
    for suffix in ("_1shot", "_2shot", "_4shot"):
        if stem.endswith(suffix):
            return suffix
    return None


def load_metadata(dataset: str, task: str = "spatio_temporal") -> dict:
    """指定 dataset の metadata を読む。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    filename = "st_test.json" if task == "spatio_temporal" else "s_test.json"
    metadata_path = Path("data") / dataset_dir / "meta-data" / filename
    return json.loads(metadata_path.read_text())


def ensure_fps1_gt_csv(dataset: str) -> None:
    """dataset-level の tubes_fps1.csv が無ければ metadata から生成する。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    target_csv = Path("data") / dataset_dir / "vidi_eval" / "vue-stg-benchmark" / "tubes_fps1.csv"
    if not target_csv.exists():
        metadata = load_metadata(dataset)
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        with target_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "time_ms", "x0", "y0", "x1", "y1"])
            for query_id, sample in metadata.items():
                fps1_tube = sample["fps1_timestamp_0_1000_st_demonstration"]["tubes"][0]["bbox"]
                for time_str, bbox in sorted(fps1_tube.items(), key=lambda item: float(item[0])):
                    time_ms = int(round(float(time_str) * 1000))
                    x0, y0, x1, y1 = bbox
                    writer.writerow(
                        [query_id, time_ms, x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]
                    )
    write_yxyx_csv_from_xyxy_csv(target_csv)


def ensure_gpt_frame_id_gt_csv(dataset: str) -> None:
    """dataset-level の GPT 用 frame_id GT CSV が無ければ metadata から生成する。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    target_csv = (
        Path("data") / dataset_dir / "vidi_eval" / "vue-stg-benchmark" / "tubes_fps1_frame_id.csv"
    )
    if not target_csv.exists():
        metadata = load_metadata(dataset)
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        with target_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "time_ms", "x0", "y0", "x1", "y1"])
            for query_id, sample in metadata.items():
                fps1_tube = sample["fps1_frame_id_0_1_demonstration"]["tubes"][0]["bbox"]
                for frame_id_str, bbox in sorted(fps1_tube.items(), key=lambda item: int(item[0])):
                    x0, y0, x1, y1 = bbox
                    writer.writerow([query_id, int(frame_id_str), x0, y0, x1, y1])


def ensure_spatial_fps1_gt_csv(dataset: str) -> None:
    """dataset-level の tubes_spatial_fps1.csv が無ければ metadata から生成する。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    target_csv = (
        Path("data") / dataset_dir / "vidi_eval" / "vue-stg-benchmark" / "tubes_spatial_fps1.csv"
    )
    if not target_csv.exists():
        metadata = load_metadata(dataset, task="spatial")
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        with target_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "time_ms", "x0", "y0", "x1", "y1"])
            for query_id, sample in metadata.items():
                fps1_tube = sample["fps1_frame_id_0_1_demonstration"]["tubes"][0]["bbox"]
                for frame_index, (_, bbox) in enumerate(
                    sorted(fps1_tube.items(), key=lambda item: int(item[0]))
                ):
                    x0, y0, x1, y1 = bbox
                    writer.writerow([query_id, frame_index * 1000, x0, y0, x1, y1])


def build_match_index(metadata: dict) -> dict:
    """query_id に依存しない照合キーから metadata と query_id を引ける index を作る。"""
    index = {}
    for query_id, sample in metadata.items():
        video_name = sample["video_name"]
        text = sample["text"]
        temporal_range = sample["temporal_range"]
        record = {"query_id": str(query_id), "sample": sample}
        index[video_name] = record
        index[f"{video_name}.mp4"] = record
        index[(video_name, temporal_range)] = record
        index[(f"{video_name}.mp4", temporal_range)] = record
        index[(video_name, text)] = record
        index[(f"{video_name}.mp4", text)] = record
    return index


def build_full_sample_index(
    metadata: dict,
    task: str = "spatio_temporal",
) -> dict[tuple[str, str, str], str]:
    """video_name + text + task固有label から query_id を引く index を作る。"""
    index = {}
    label_key = "spatio_temporal_label" if task == "spatio_temporal" else "spatial_label"
    for query_id, sample in metadata.items():
        key = (
            sample["video_name"],
            sample["text"],
            json.dumps(sample[label_key], sort_keys=True, ensure_ascii=False),
        )
        index[key] = str(query_id)
    return index


def resolve_dataset_from_domain_query_id(
    domain_query_id: str,
    domain_metadata: dict,
    candidate_full_indices: dict[str, dict[tuple[str, str, str], str]],
    task: str = "spatio_temporal",
) -> tuple[str, str]:
    """domain metadata 上の query_id から元 dataset と dataset 内 query_id を解決する。"""
    sample = domain_metadata[str(domain_query_id)]
    label_key = "spatio_temporal_label" if task == "spatio_temporal" else "spatial_label"
    key = (
        sample["video_name"],
        sample["text"],
        json.dumps(sample[label_key], sort_keys=True, ensure_ascii=False),
    )
    matched = [
        (dataset, full_index[key])
        for dataset, full_index in candidate_full_indices.items()
        if key in full_index
    ]
    if len(matched) == 1:
        return matched[0]
    if not matched:
        raise KeyError(f"Cannot map domain query_id={domain_query_id} to dataset metadata")
    raise ValueError(f"Ambiguous dataset mapping for domain query_id={domain_query_id}: {matched}")


def resolve_spatial_dataset_from_domain_query_id(
    domain_query_id: str,
    domain_metadata: dict,
) -> tuple[str, str]:
    """spatial 2shot の domain query_id から元 dataset と dataset 内 query_id を解決する。"""
    sample = domain_metadata[str(domain_query_id)]
    video_name = str(sample["video_name"])
    dataset_prefixes = [
        ("animal_kingdom", "animal_kingdom"),
        ("american_football", "american_football"),
        ("CholecTrack20", "cholectrack20"),
        ("egosurgery", "egosurgery"),
        ("MECCANO", "meccano"),
        ("MultiSports", "multisports"),
        ("ENIGMA", "enigma"),
        ("DoTA", "dota"),
        ("mouse", "mouse"),
        ("uca", "uca"),
    ]
    for prefix, dataset in dataset_prefixes:
        needle = f"{prefix}_"
        if video_name.startswith(needle):
            return dataset, video_name[len(needle):]
    raise KeyError(f"Cannot map spatial domain query_id={domain_query_id} via video_name={video_name}")


def find_dataset_for_item(item: dict, candidate_indices: dict[str, dict]) -> tuple[str, str]:
    """1 prediction item の所属 dataset と dataset 内 query_id を metadata から判定する。"""
    gt = item.get("gt") or item.get("ground_truth")
    gt_temporal_range = None if gt is None else gt.get("temporal_range")
    if isinstance(gt_temporal_range, list):
        gt_temporal_range = " ".join(str(v) for v in gt_temporal_range)
    query_video_path = item.get("query_video_path") or item.get("query")
    text = item.get("text")
    video_stem = Path(query_video_path).stem
    candidate_keys = [
        query_video_path,
        video_stem,
        (query_video_path, gt_temporal_range),
        (video_stem, gt_temporal_range),
        (query_video_path, text),
        (video_stem, text),
    ]

    matched_records = []
    for dataset, index in candidate_indices.items():
        for key in candidate_keys:
            if key in index:
                matched_records.append((dataset, index[key]["query_id"]))
                break

    if len(matched_records) == 1:
        return matched_records[0]
    if not matched_records:
        raise KeyError(f"Cannot match prediction item to dataset: {item}")
    raise ValueError(f"Ambiguous dataset match {matched_records} for item: {item}")


def split_predictions_by_dataset(pred_json: Path) -> dict[str, list[dict]]:
    """Prediction JSON を domain 内の dataset ごとに分解する。"""
    target_type, target = infer_prediction_target(pred_json)
    predictions = json.loads(pred_json.read_text())
    if target_type == "domain" or get_shot_suffix(pred_json.stem) is None:
        return {target: predictions}

    domain = DATASET_TO_DOMAIN[target]
    candidate_datasets = [
        dataset for row_domain, dataset in DOMAIN_DATASET_ORDER if row_domain == domain
    ]
    if len(candidate_datasets) == 1:
        return {candidate_datasets[0]: predictions}

    candidate_indices = {
        dataset: build_match_index(load_metadata(dataset))
        for dataset in candidate_datasets
    }
    candidate_full_indices = {
        dataset: build_full_sample_index(load_metadata(dataset), task="spatio_temporal")
        for dataset in candidate_datasets
    }
    domain_metadata = load_metadata(domain)
    split_predictions: dict[str, list[dict]] = {dataset: [] for dataset in candidate_datasets}
    for item in predictions:
        dataset, dataset_query_id = resolve_dataset_from_domain_query_id(
            domain_query_id=str(item["query_id"]),
            domain_metadata=domain_metadata,
            candidate_full_indices=candidate_full_indices,
            task="spatio_temporal",
        )
        normalized_item = dict(item)
        normalized_item["query_id"] = dataset_query_id
        split_predictions[dataset].append(normalized_item)
    return {dataset: items for dataset, items in split_predictions.items() if items}


def split_predictions_by_dataset_for_task(
    pred_json: Path,
    task: str,
) -> dict[str, list[dict]]:
    """task に応じて Prediction JSON を dataset ごとに分解する。"""
    target_type, target = infer_prediction_target(pred_json)
    predictions = json.loads(pred_json.read_text())
    if target_type == "domain" or get_shot_suffix(pred_json.stem) is None:
        return {target: predictions}

    if task == "spatio_temporal":
        return split_predictions_by_dataset(pred_json)

    domain = DATASET_TO_DOMAIN[target]
    candidate_datasets = [
        dataset for row_domain, dataset in DOMAIN_DATASET_ORDER if row_domain == domain
    ]
    if len(candidate_datasets) == 1:
        return {candidate_datasets[0]: predictions}

    domain_metadata = load_metadata(domain, task=task)
    split_predictions: dict[str, list[dict]] = {dataset: [] for dataset in candidate_datasets}
    for item in predictions:
        if task == "spatio_temporal":
            candidate_full_indices = {
                dataset: build_full_sample_index(load_metadata(dataset, task=task), task=task)
                for dataset in candidate_datasets
            }
            dataset, dataset_query_id = resolve_dataset_from_domain_query_id(
                domain_query_id=str(item["query_id"]),
                domain_metadata=domain_metadata,
                candidate_full_indices=candidate_full_indices,
                task=task,
            )
        else:
            dataset, dataset_query_id = resolve_spatial_dataset_from_domain_query_id(
                domain_query_id=str(item["query_id"]),
                domain_metadata=domain_metadata,
            )
        normalized_item = dict(item)
        normalized_item["query_id"] = dataset_query_id
        split_predictions[dataset].append(normalized_item)
    return {dataset: items for dataset, items in split_predictions.items() if items}


def load_summary_rows(summary_csv: Path) -> list[dict]:
    """summary.csv を読み込んで行 dict の list を返す。"""
    with summary_csv.open() as f:
        return list(csv.DictReader(f))


def _json_safe_value(value):
    if value != value:
        return None
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    return str(value)


def _row_to_metrics(row: dict) -> dict:
    metrics = {
        key: _json_safe_value(row[key])
        for key in [
            "t_Precision",
            "t_Recall",
            "t_IoU",
            "v_Precision",
            "v_Recall",
            "v_IoU",
            "v_IoU_Int",
        ]
        if key in row
    }
    threshold_sources = {
        "t_IoU": ["t_IoU@0.3", "t_IoU@0.5", "t_IoU@0.7"],
        "v_IoU": ["v_IoU@0.3", "v_IoU@0.5", "v_IoU@0.7"],
        "v_IoU_Int": ["v_IoU_Int@0.3", "v_IoU_Int@0.5", "v_IoU_Int@0.7"],
    }
    for source_key, target_keys in threshold_sources.items():
        if source_key not in row or row[source_key] != row[source_key]:
            continue
        source_value = float(row[source_key])
        for threshold, target_key in zip([0.3, 0.5, 0.7], target_keys):
            metrics[target_key] = float(source_value >= threshold)
    return metrics


def build_prediction_entries(source_json: Path, task: str) -> list[dict]:
    """元 prediction JSON の各要素に dataset と解決済み query_id を付ける。"""
    predictions = json.loads(source_json.read_text())
    target_type, target = infer_prediction_target(source_json)

    if target_type == "domain" or get_shot_suffix(source_json.stem) is None:
        return [
            {
                "source_index": index,
                "dataset": target,
                "resolved_query_id": str(item["query_id"]),
                "item": item,
            }
            for index, item in enumerate(predictions)
        ]

    domain = DATASET_TO_DOMAIN[target]
    candidate_datasets = [
        dataset for row_domain, dataset in DOMAIN_DATASET_ORDER if row_domain == domain
    ]
    domain_metadata = load_metadata(domain, task=task)
    entries = []
    for index, item in enumerate(predictions):
        if task == "spatio_temporal":
            candidate_full_indices = {
                dataset: build_full_sample_index(load_metadata(dataset, task=task), task=task)
                for dataset in candidate_datasets
            }
            dataset, dataset_query_id = resolve_dataset_from_domain_query_id(
                domain_query_id=str(item["query_id"]),
                domain_metadata=domain_metadata,
                candidate_full_indices=candidate_full_indices,
                task=task,
            )
        else:
            dataset, dataset_query_id = resolve_spatial_dataset_from_domain_query_id(
                domain_query_id=str(item["query_id"]),
                domain_metadata=domain_metadata,
            )
        entries.append(
            {
                "source_index": index,
                "dataset": dataset,
                "resolved_query_id": str(dataset_query_id),
                "item": item,
            }
        )
    return entries


def build_per_query_metrics(
    dataset: str,
    pred_csv_path: Path,
    step_ms: int,
    ignore_missing_pred: bool,
    task: str,
) -> dict[str, dict]:
    """dataset 内 query_id ごとの full_fps/fps1 metric を返す。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    gt_dir = Path("data") / dataset_dir / "vidi_eval" / "vue-stg-benchmark"
    evaluator = SpatioTemporalEvaluator(step_ms=step_ms)
    metrics_by_query_id: dict[str, dict] = {}

    if task == "spatial":
        variants = [("spatial_fps1", "tubes_spatial_fps1.csv")]
    else:
        variants = [("full_fps", "tubes.csv"), ("fps1", "tubes_fps1.csv")]

    for variant, tubes_file in variants:
        gt_tubes_path = gt_dir / tubes_file
        if not gt_tubes_path.exists():
            continue
        evaluator.load_dataset(str(gt_dir), tubes_file=tubes_file)
        df = evaluator.evaluate_pred_file(
            str(pred_csv_path),
            grouped=True,
            ignore_missing_pred=ignore_missing_pred,
        )
        for row in df.to_dict(orient="records"):
            query_id = str(row["query_id"])
            metrics_by_query_id.setdefault(query_id, {})[variant] = _row_to_metrics(row)

    return metrics_by_query_id


def enrich_gt_with_metadata(dataset: str, resolved_query_id: str, item: dict, task: str) -> dict:
    """metadata 上の GT 情報を item['gt'] に寄せた dict を返す。"""
    metadata = load_metadata(dataset, task=task)
    sample = metadata[resolved_query_id]
    raw_gt = item.get("gt") or {}
    if isinstance(raw_gt, str):
        try:
            gt = json.loads(raw_gt)
        except json.JSONDecodeError:
            gt = {}
    else:
        gt = dict(raw_gt)
    payload_keys = GT_PAYLOAD_KEYS if task == "spatio_temporal" else SPATIAL_GT_PAYLOAD_KEYS
    for key in payload_keys:
        if key in sample:
            gt[key] = sample[key]
    return gt


def get_query_text(dataset: str, resolved_query_id: str, task: str) -> str:
    """metadata 上の query text を返す。"""
    metadata = load_metadata(dataset, task=task)
    return str(metadata[resolved_query_id]["text"])


def write_per_query_augmented_json(
    source_json: Path,
    output_dir: Path,
    per_index_payload: dict[int, dict],
    prediction_entries: list[dict] | None = None,
    task: str = "spatio_temporal",
) -> list[Path]:
    """元 prediction JSON に metric/gt を付与した派生 JSON を保存する。"""
    original_items = json.loads(source_json.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_suffix = get_shot_suffix(source_json.stem)
    if shot_suffix is not None and prediction_entries is not None:
        items_by_dataset: dict[str, list[dict]] = {}
        timestamp = source_json.stem.split("_", 2)[:2]
        timestamp_prefix = "_".join(timestamp)
        for entry in prediction_entries:
            index = entry["source_index"]
            augmented_item = dict(original_items[index])
            augmented_item["dataset"] = entry["dataset"]
            augmented_item["query_id"] = entry["resolved_query_id"]
            augmented_item["resolved_query_id"] = entry["resolved_query_id"]
            augmented_item.update(per_index_payload[index])
            items_by_dataset.setdefault(entry["dataset"], []).append(augmented_item)

        output_paths = []
        for dataset, augmented_items in items_by_dataset.items():
            output_path = output_dir / (
                f"{timestamp_prefix}_{dataset}{shot_suffix}_with_per_query_metrics_{task}.json"
            )
            output_path.write_text(
                json.dumps(augmented_items, ensure_ascii=False, indent=2) + "\n"
            )
            output_paths.append(output_path)
        return output_paths

    augmented_items = []
    for index, item in enumerate(original_items):
        augmented_item = dict(item)
        augmented_item.update(per_index_payload[index])
        augmented_items.append(augmented_item)

    output_path = output_dir / f"{source_json.stem}_with_per_query_metrics_{task}.json"
    output_path.write_text(json.dumps(augmented_items, ensure_ascii=False, indent=2) + "\n")
    return [output_path]


def run_single(
    model: str,
    source_json: Path,
    step_ms: int,
    ignore_missing_pred: bool,
    per_query_output_dir: Path | None = None,
    task: str = "spatio_temporal",
) -> list[dict]:
    """1 prediction JSON を dataset ごとに分解して変換・評価する。"""
    dataset_predictions = split_predictions_by_dataset_for_task(source_json, task=task)
    prediction_entries = build_prediction_entries(source_json, task=task)
    bundle_dir = Path("data/eval") / model / f"{task}_bundle_inputs"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shot_suffix = get_shot_suffix(source_json.stem)
    stem_base = source_json.stem
    if shot_suffix is not None:
        stem_base = stem_base[: -len(shot_suffix)]

    rows = []
    per_index_payload: dict[int, dict] = {}
    for dataset, predictions in dataset_predictions.items():
        dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
        split_pred_json = bundle_dir / f"{stem_base}_{dataset}.json"
        split_pred_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")

        target_csv = Path("data") / dataset_dir / "vidi_eval" / "results" / f"{model}_{task}" / "tubes.csv"
        print(f"Source JSON: {source_json}")
        print(f"Dataset: {dataset}")
        print(f"Eval target: {dataset_dir}")
        print(f"Target CSV: {target_csv}")

        convert_prediction_json_to_vidi_tubes(
            split_pred_json,
            target_csv,
            model=model,
            task=task,
        )
        if task == "spatio_temporal" and model == "gpt":
            ensure_gpt_frame_id_gt_csv(dataset)
        elif task == "spatio_temporal":
            ensure_fps1_gt_csv(dataset)
        else:
            ensure_spatial_fps1_gt_csv(dataset)
        summary_csv = Path(
            run_vidi_evaluation(
                dataset=dataset_dir,
                model=model,
                step_ms=step_ms,
                ignore_missing_pred=ignore_missing_pred,
                task=task,
            )
        )
        metrics_by_query_id = build_per_query_metrics(
            dataset=dataset,
            pred_csv_path=target_csv,
            step_ms=step_ms,
            ignore_missing_pred=ignore_missing_pred,
            task=task,
        )
        for entry in prediction_entries:
            if entry["dataset"] != dataset:
                continue
            per_index_payload[entry["source_index"]] = {
                "text": get_query_text(
                    dataset=dataset,
                    resolved_query_id=entry["resolved_query_id"],
                    task=task,
                ),
                "text_query": get_query_text(
                    dataset=dataset,
                    resolved_query_id=entry["resolved_query_id"],
                    task=task,
                ),
                "gt": enrich_gt_with_metadata(
                    dataset=dataset,
                    resolved_query_id=entry["resolved_query_id"],
                    item=entry["item"],
                    task=task,
                ),
                "metric": metrics_by_query_id.get(entry["resolved_query_id"], {}),
            }
        n = len(predictions)
        for summary_row in load_summary_rows(summary_csv):
            rows.append(
                {
                    "model": model,
                    "domain": domain_for_target(dataset),
                    "row_type": "domain" if is_domain_target(dataset) else "dataset",
                    "dataset": "ALL" if is_domain_target(dataset) else dataset,
                    "n": n,
                    **summary_row,
                }
            )
    if per_query_output_dir is not None:
        output_paths = write_per_query_augmented_json(
            source_json=source_json,
            output_dir=per_query_output_dir,
            per_index_payload=per_index_payload,
            prediction_entries=prediction_entries,
            task=task,
        )
        for output_path in output_paths:
            print(f"saved_per_query={output_path}")
    return rows


def domain_weighted_rows(dataset_rows: list[dict]) -> list[dict]:
    """同一 domain 内の dataset 行を件数重み付きで集約する。"""
    dataset_rows = [row for row in dataset_rows if row["row_type"] != "domain"]
    if not dataset_rows:
        return []
    metric_keys = [
        key for key in dataset_rows[0].keys()
        if key not in {"model", "domain", "row_type", "dataset", "n", *META_KEYS}
    ]
    output = []
    for domain in dict.fromkeys(domain for domain, _ in DOMAIN_DATASET_ORDER):
        domain_rows = [row for row in dataset_rows if row["domain"] == domain]
        if not domain_rows:
            continue
        combos = {
            tuple(row[key] for key in META_KEYS)
            for row in domain_rows
        }
        for combo in combos:
            matched_rows = [
                row for row in domain_rows
                if tuple(row[key] for key in META_KEYS) == combo
            ]
            weighted = {}
            combo_n = sum(
                row["n"] for dataset in {row["dataset"] for row in matched_rows}
                for row in matched_rows
                if row["dataset"] == dataset
            )
            for metric in metric_keys:
                valid_rows = [
                    row for row in matched_rows
                    if row.get(metric) not in ("", None)
                ]
                if not valid_rows:
                    weighted[metric] = ""
                    continue
                valid_n = sum(row["n"] for row in valid_rows)
                weighted[metric] = (
                    sum(float(row[metric]) * row["n"] for row in valid_rows) / valid_n
                )
            output.append(
                {
                    "model": domain_rows[0]["model"],
                    "domain": domain,
                    "row_type": "domain_weighted",
                    "dataset": "ALL",
                    "n": combo_n,
                    **{key: value for key, value in zip(META_KEYS, combo)},
                    **weighted,
                }
            )
    return output


def ordered_summary_rows(dataset_rows: list[dict]) -> list[dict]:
    """Dataset 行と domain 集約行を所定順で並べる。overall 行を先頭に置く。"""
    weighted_rows = domain_weighted_rows(dataset_rows)
    overall_rows = []
    detailed_rows = []
    for domain in dict.fromkeys(domain for domain, _ in DOMAIN_DATASET_ORDER):
        direct_domain_rows = [
            row for row in dataset_rows
            if row["domain"] == domain and row["row_type"] == "domain"
        ]
        if direct_domain_rows:
            overall_rows.extend(
                row for row in direct_domain_rows
                if row["group"] == "overall" and row["category"] == "overall"
            )
            detailed_rows.extend(
                row for row in direct_domain_rows
                if not (row["group"] == "overall" and row["category"] == "overall")
            )
            continue
        matched_weighted_rows = [row for row in weighted_rows if row["domain"] == domain]
        overall_rows.extend(
            row for row in matched_weighted_rows
            if row["group"] == "overall" and row["category"] == "overall"
        )
        for _row_domain, dataset in DOMAIN_DATASET_ORDER:
            if _row_domain != domain:
                continue
            matched_dataset_rows = [
                row for row in dataset_rows
                if row["domain"] == domain and row["dataset"] == dataset
            ]
            overall_rows.extend(
                row for row in matched_dataset_rows
                if row["group"] == "overall" and row["category"] == "overall"
            )
            detailed_rows.extend(
                row for row in matched_dataset_rows
                if not (row["group"] == "overall" and row["category"] == "overall")
            )
        detailed_rows.extend(
            row for row in matched_weighted_rows
            if not (row["group"] == "overall" and row["category"] == "overall")
        )
    return overall_rows + detailed_rows


def write_bundle_summary(base_path: Path, rows: list[dict]) -> None:
    """bundle 実行結果を variant ごとの summary.csv に分けて保存する。"""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])

    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if len(variants) == 1 and variant == "spatial_fps1":
            variant_path = base_path
        else:
            variant_path = base_path.with_name(f"{base_path.stem}_{variant}{base_path.suffix}")
        fieldnames = list(variant_rows[0].keys())
        with variant_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            split_idx = next(
                (
                    idx
                    for idx, row in enumerate(variant_rows)
                    if not (row["group"] == "overall" and row["category"] == "overall")
                ),
                len(variant_rows),
            )
            writer.writerows(variant_rows[:split_idx])
            if split_idx < len(variant_rows):
                writer.writerow({})
                writer.writerows(variant_rows[split_idx:])


def main() -> None:
    """複数の spatio-temporal prediction JSON を dataset ごとに評価して集約する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["spatio_temporal", "spatial"], default="spatio_temporal")
    parser.add_argument("--pred-jsons", nargs="+", required=True)
    parser.add_argument("--summary-csv")
    parser.add_argument("--step-ms", type=int, default=1000)
    parser.add_argument("--ignore-missing-pred", action="store_true")
    parser.add_argument("--per-query-output-dir")
    args = parser.parse_args()

    default_outputs_dir = Path("data/eval") / args.model / "outputs"
    per_query_output_dir = (
        Path(args.per_query_output_dir)
        if args.per_query_output_dir
        else default_outputs_dir
    )
    dataset_rows = []
    for pred_json in [Path(path) for path in args.pred_jsons]:
        dataset_rows.extend(
            run_single(
                model=args.model,
                source_json=pred_json,
                step_ms=args.step_ms,
                ignore_missing_pred=args.ignore_missing_pred,
                per_query_output_dir=per_query_output_dir,
                task=args.task,
            )
        )

    summary_rows = ordered_summary_rows(dataset_rows)
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path("data/eval") / args.model / "outputs" / f"{args.task}_summary.csv"
    )
    write_bundle_summary(summary_csv, summary_rows)
    print(f"saved = {summary_csv}")


if __name__ == "__main__":
    main()
