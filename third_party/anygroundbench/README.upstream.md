# AnyGroundBench

<p align="center">
  <a href="https://arxiv.org/abs/2607.02269"><img src="https://img.shields.io/badge/arXiv-2607.02269-b31b1b.svg" alt="arXiv"></a>
  <a href="https://rinost081.github.io/AnyGroundBench-page/"><img src="https://img.shields.io/badge/Project%20Page-online-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/rinost081/AnyGroundBench"><img src="https://img.shields.io/badge/Hugging%20Face-dataset-yellow.svg" alt="Hugging Face"></a>
</p>

<div align="center">

**AnyGroundBench: A Specialized-Domain Benchmark for Video Grounding in Vision-Language Models**

Rintaro Otsubo\*<sup>1,2</sup> · Ryo Fujii\*<sup>1,2</sup> · Reina Ishikawa<sup>1,2</sup> · Taiki Kanaya<sup>1,2</sup> · Kanta Sawafuji<sup>1,2</sup><br>
Hiroki Kajita<sup>1,3</sup> · Shigeki Sakai<sup>1,3</sup> · Hideo Saito<sup>1,2</sup> · Ryo Hachiuma<sup>4</sup>

<sub>
<sup>1</sup> Keio University &nbsp;
<sup>2</sup> Keio AI Research Center &nbsp;
<sup>3</sup> Keio University School of Medicine &nbsp;
<sup>4</sup> NVIDIA
</sub>

<sub>\* Equal contribution.</sub>

</div>

AnyGroundBench is a domain-adaptation benchmark for evaluating video grounding in vision-language models across specialized domains with dense temporal, spatial, and spatio-temporal annotations.

## Overview

AnyGroundBench evaluates how vision-language models adapt to specialized-domain video grounding across animal, industry, sports, surgery, and public-security scenarios using unified temporal, spatial, and spatio-temporal annotations.

## Datasets

AnyGroundBench is organized by domain at runtime.
Each domain is built from one or more source datasets:

| Dataset | Source type | Domain | Videos / source |
| --- | --- | --- | --- |
| `mouse` | Newly captured | Animal | [Hugging Face](https://huggingface.co/datasets/rinost081/AnyGroundBench) |
| `animal_kingdom` | Existing dataset | Animal | [Original source](https://sutdcv.github.io/Animal-Kingdom/) |
| `enigma` | Existing dataset | Industry | [Original source](https://iplab.dmi.unict.it/ENIGMA-360/) |
| `meccano` | Existing dataset | Industry | [Original source](https://iplab.dmi.unict.it/MECCANO/) |
| `dota` | Existing dataset | Public security | [Original source](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) |
| `uca` | Existing dataset | Public security | [Original source](https://www.crcv.ucf.edu/projects/real-world/) |
| `american_football` | Newly captured | Sports | [Hugging Face](https://huggingface.co/datasets/rinost081/AnyGroundBench) |
| `multisports` | Existing dataset | Sports | [Original source](https://github.com/MCG-NJU/MultiSports) |
| `cholectrack20` | Existing dataset | Surgery | [Original source](https://github.com/CAMMA-public/cholectrack20) |
| `egosurgery` | Existing dataset | Surgery | [Original source](https://github.com/Fujiry0/EgoSurgery) |

The source dataset names describe where the videos come from. The benchmark runtime reads the merged domain-level data under `data/<domain_name>/`.

## Data Preparation

The runtime layout is domain-level:

```text
data/<domain_name>/
├── meta-data
│   ├── t_train.json
│   ├── t_test.json
│   ├── s_train.json
│   ├── s_test.json
│   ├── st_train.json
│   └── st_test.json
├── simrank
│   └── video_top100_alpha<alpha>.json
└── videos
    ├── clips
    │   └── <video_name>.mp4
    └── clips4spatial
        └── <clip_name>.mp4
```

The `meta-data/` and `simrank/` files are included in the AnyGroundBench Hugging Face release. Source videos must be reconstructed into `data/<domain_name>/videos/` using the domain-specific preparation scripts.

Follow the corresponding preparation guide:

| Domain | Preparation guide |
| --- | --- |
| `animal` | [`create_anygroundbench/animal/README.md`](create_anygroundbench/animal/README.md) |
| `industry` | [`create_anygroundbench/industry/README.md`](create_anygroundbench/industry/README.md) |
| `safety` | [`create_anygroundbench/safety/README.md`](create_anygroundbench/safety/README.md) |
| `sports` | [`create_anygroundbench/sports/README.md`](create_anygroundbench/sports/README.md) |
| `surgery` | [`create_anygroundbench/surgery/README.md`](create_anygroundbench/surgery/README.md) |

General workflow:

1. Download the AnyGroundBench release files under `data/<domain_name>/`.
2. Download the required original source videos for the target domain.
3. Follow the corresponding `create_anygroundbench/<domain_name>/README.md` guide above to reconstruct `videos/clips`.
4. Run the domain spatial clip script to create `videos/clips4spatial`.

Do not use `data/<source_dataset>/` as the runtime input. Source dataset directories such as `data/animal_kingdom/`, `data/MECCANO/`, or `data/DoTA/` are raw inputs for the reconstruction scripts. The benchmark runtime reads `data/<domain_name>/...`.

## Inference

Run inference with `test.py` by specifying a task, domain, and model.

For API-based models, copy [`.env.example`](.env.example) to `.env` and set credentials:

```bash
GEMINI_API_KEY=<your_gemini_api_key>
```

Gemini 2.5 Flash zero-shot:

```bash
uv run --project envs/gemini python test.py --task temporal --domain_name animal --model_name gemini --model_id gemini-2.5-flash --n_shot 0
```

Gemini 2.5 Flash 2-shot:

```bash
uv run --project envs/gemini python test.py --task temporal --domain_name animal --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5
```

Qwen-3VL-8B zero-shot:

```bash
uv run --project envs/qwen python test.py --task temporal --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 0
```

Qwen-3VL-8B 2-shot:

```bash
uv run --project envs/qwen python test.py --task temporal --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5
```

See [`scripts/inference_0shot.sh`](scripts/inference_0shot.sh) and [`scripts/inference_2shot.sh`](scripts/inference_2shot.sh) for full command lists across tasks, domains, and models.

## Evaluation

Evaluation examples are provided in `scripts/eval/`.

The evaluation scripts infer the domain from each prediction filename. Use domain suffixes such as
`animal`, `industry`, `sports`, `surgery`, or `safety` in the prediction JSON filename.
For example, evaluate an animal-domain inference output as `results/<task>/<timestamp>_animal.json`.

Temporal Grounding evaluation examples
```text
# Gemini 2.5 Flash temporal grounding
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py \
  --model gemini-2.5-flash \
  --pred-jsons results/temporal/<timestamp>_animal.json

# Qwen-3VL-8B temporal grounding
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py \
  --model qwen3-vl-8b \
  --pred-jsons results/temporal/<timestamp>_animal.json
```

Spatial grounding evaluation examples
```text
# Gemini 2.5 Flash spatial grounding
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py \
  --task spatial \
  --model gemini-2.5-flash \
  --pred-jsons results/spatial/<timestamp>_animal.json

# Qwen-3VL-8B spatial grounding
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py \
  --task spatial \
  --model qwen3-vl-8b \
  --pred-jsons results/spatial/<timestamp>_animal.json
```
Spatio-Temporal Grounding evaluation examples

```text
# Gemini 2.5 Flash spatio-temporal grounding
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py \
  --task spatio_temporal \
  --model gemini-2.5-flash \
  --pred-jsons results/spatio_temporal/<timestamp>_animal.json

# Qwen-3VL-8B spatio-temporal grounding
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py \
  --task spatio_temporal \
  --model qwen3-vl-8b \
  --pred-jsons results/spatio_temporal/<timestamp>_animal.json
```

## License

The newly captured `mouse` and `american_football` data are under CC BY-NC-SA 4.0.

Third-party datasets remain governed by their original licenses and terms. Users are responsible for complying with the original dataset providers' licenses, data use agreements, and redistribution restrictions.

## Citation

```bibtex
@misc{otsubo2026anygroundbenchspecializeddomainbenchmarkvideo,
      title={AnyGroundBench: A Specialized-Domain Benchmark for Video Grounding in Vision-Language Models},
      author={Rintaro Otsubo and Ryo Fujii and Reina Ishikawa and Taiki Kanaya and Kanta Sawafuji and Hiroki Kajita and Shigeki Sakai and Hideo Saito and Ryo Hachiuma},
      year={2026},
      eprint={2607.02269},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.02269},
}
```

## Contact

