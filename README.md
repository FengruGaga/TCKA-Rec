# TCKA-Rec

Task-Conditioned Knowledge Augmentation for Extensible Multi-Task Recommendation.

Implementation of the manuscript **“基于任务条件知识增强的可扩展多任务推荐方法”** (Journal of Software, under review). The code trains a frozen multi-task backbone with a self-supervised universal expert `U`, a conditional low-rank readout for original tasks, and task-private LoRA adaptation for newly added targets.

This repository is a clean source release. It contains the method, the Table 2 baselines, reproduction configs, and the recorded seeds. It does **not** contain datasets, checkpoints, prediction files, logs, or paper drafts.

## Repository layout

```text
tcka_rec/                 # Method and training pipeline
  cli.py                  # New-task experiments (Tables 3 and 4)
  table2_pilot.py         # Validation-only Table 2 pilot
  table2_runner.py        # Table 2 formal queue and report
  table2_data.py          # Census/Ali-CCP two-task data admission
  table2_backbone.py      # Two-task backbone and U source
  baseline_models.py      # SharedBottom, MMoE, PLE, AdaTT
  vendor/                 # MPT-Rec baseline classes and AdaTT (MIT)
  single_task.py          # Independent Single Task baseline
  readout_models.py       # Conditional low-rank readout
  readout_fitting.py      # Readout fitting and raw-result checks
configs/                  # Dataset, training, and seed configuration
tests/                    # CPU-level unit and protocol tests
train.py                  # Entry point for new-task experiments
```

## Environment

Python 3.10+ and PyTorch 2.1+ are recommended; the paper used PyTorch 2.1.1 with CUDA 11.8.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

A matching Conda environment is provided in `environment.yml`. The Table 2 queue uses Linux `fork`; run it on Linux or WSL. `train.py` and the unit tests also run on Windows.

## Data

No data are bundled. The released code expects the following layout:

```text
dataset/
  Census-income/
    raw/census-income.data
    raw/census-income.test
    train.gz
    test.gz
  AliCCP/
    ctr_cvr.train
    ctr_cvr.dev
    ctr_cvr.test
  KuaiRand-Pure/
    data/log_standard_4_08_to_4_21_pure.csv
    data/log_standard_4_22_to_5_08_pure.csv
    data/log_random_4_22_to_5_08_pure.csv
```

### Census-income

- Source: [UCI Census-Income (KDD)](https://archive.ics.uci.edu/dataset/117/census+income+kdd).
- Download `census-income.data` and `census-income.test` into `dataset/Census-income/raw/`.
- Produce the encoded `train.gz` and `test.gz` with the preprocessing script from [BAI-LAB/MPT-Rec](https://github.com/BAI-LAB/MPT-Rec/blob/two_task/multitaskrec/data_preprocessing/CensusIncome_process.py). The script exposes `process(data_path, write_path)`; place its two outputs next to `raw/`.
- Expected SHA256:

```text
e2f2d57925a8020d23fcd05802284b3bc1d91fcdd451f1eaaefd683e76d1b2dc  Census-income/train.gz
8de690375028ac531de37fcbb117bd6527c5f3542d3bfead8975226e45100cec  Census-income/test.gz
```

The original test file is split equally into validation and test with seed `20260929`.

### Ali-CCP

- Source: [Tianchi Ali-CCP](https://tianchi.aliyun.com/dataset/408).
- Download `sample_skeleton_{train,test}.csv` and `common_features_{train,test}.csv`.
- Produce `ctr_cvr.train`, `ctr_cvr.dev` and `ctr_cvr.test` with the preprocessing script from [BAI-LAB/MPT-Rec](https://github.com/BAI-LAB/MPT-Rec/blob/two_task/multitaskrec/data_preprocessing/AliCCP_process.py). The script expects the files under `dataset/AliCCP/`.
- Expected SHA256:

```text
707d8d264d2e46fe4194419dfd9cb3bad09b5ade1a05e9a402e8a3273332e216  AliCCP/ctr_cvr.train
d2df3dc96dca7b6cc1e35284ee22015cb0265e1e4ac85ae8654eb63cde6dc727  AliCCP/ctr_cvr.dev
11f997fc9856f3392f0fe21ebf4119fc7c65e9ac09023f440a9ce1054b679633  AliCCP/ctr_cvr.test
```

The protocol reads the first 10M training, 1M validation, and 10M test rows.

### KuaiRand-Pure

- Source: [KuaiRand-Pure](https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz), archive MD5 `0820331067a3784d9691136f772b35a7`.
- Extract the archive so that the three CSV files are under `dataset/KuaiRand-Pure/data/`.
- The released code fits the categorical vocabulary on the training split and writes its own cache under `cache/kuairand`. Input SHA256 values are in `configs/kuairand.json`.

## Reproduce Table 2 (original tasks)

Table 2 compares Single Task, SharedBottom, MMoE, PLE, AdaTT, MPT-Rec and TCKA on the two original tasks of each dataset. The formal seeds are in `configs/table2.json`; the pilot uses seed `1501773270`.

Run the validation-only pilot first:

```bash
python -m tcka_rec.table2_pilot \
  --config configs/table2.json \
  --dataset-root dataset \
  --cache-root cache \
  --output outputs/table2-pilot
```

Then run the formal queue:

```bash
python -m tcka_rec.table2_runner \
  --config configs/table2.json \
  --dataset-root dataset \
  --cache-root cache \
  --output outputs/table2 \
  --pilot-root outputs/table2-pilot \
  --l2-modes census=sum,aliccp=sum,kuairand=mean \
  --device cuda:0
```

`--l2-modes` records the embedding-L2 form used in the paper. If the pilot returns a `review` decision, the explicit modes are accepted as the formal decision; otherwise the pilot must pass. Adjust `execution.start_workers`, `max_workers` and `max_aliccp_io` in `configs/table2.json` to your GPU/CPU memory.

The completed run writes `metrics-mean-variance.json` and `summary.json` under `outputs/table2`. Each model/seed directory also keeps its predictions and validation curve for auditing. An optional `--source-root` can point to a previously admitted source tree; omit it for a fresh run.

## Reproduce Tables 3 and 4 (new tasks)

The new-task configs use five paired seeds and train MPT-Rec (`no_u`) and TCKA (`tcka`) by default:

```bash
python train.py --config configs/census.json \
  --dataset-root dataset --output outputs/census
python train.py --config configs/aliccp.json \
  --dataset-root dataset --output outputs/aliccp
python train.py --config configs/kuairand.json \
  --dataset-root dataset --cache-root cache --output outputs/kuairand
```

The ablation variants are selected with `--models`:

| Model key | Meaning |
|---|---|
| `no_u` | MPT-Rec path, U branch removed |
| `random_u` | random U source (TCKA − SSL) |
| `fixed_fusion` | frozen pretrained U without LoRA (TCKA − Adapt) |
| `tcka` | full TCKA |
| `single_task` | independently trained Single Task baseline |

Example:

```bash
python train.py --config configs/census.json \
  --dataset-root dataset --output outputs/census-ablation \
  --models no_u random_u fixed_fusion tcka single_task
```

Each run writes checkpoints, metrics, and raw predictions under `outputs/<dataset>/<seed>/`. Validation AUC selects the checkpoint; patience is 5.

## Seeds

- Table 2 formal five paired seeds: `2050630552`, `958075839`, `1428735616`, `1452179956`, `1019741749`.
- Table 2 pilot: `1501773270`. The paper also records `1843815127` as a validation-tuning seed; the released pilot protocol uses `1501773270`.
- New tasks on Census-income and KuaiRand-Pure: `1685480945`, `1685463909`, `1685477428`, `1685459668`, `1685496394`.
- New tasks on Ali-CCP: `1688723512`, `1688723740`, `1688738016`, `1688749593`, `1688762746`.
- Census holdout split seed: `20260929`.
- U masking RNG seeds: `20261009` (training) and `20261026` (validation masks).

## Baselines and attribution

`SharedBottom`, `MMoE`, `PLE` and the MPT-Rec backbone are adapted from [BAI-LAB/MPT-Rec](https://github.com/BAI-LAB/MPT-Rec) with local reproduction fixes. `AdaTT` is the official AdaTT-sp implementation from [facebookresearch/AdaTT](https://github.com/facebookresearch/AdaTT) at commit `0efdd821179029bbd599043e353841c1afd3fcfa`, preserved under its MIT license in `tcka_rec/vendor/LICENSE.AdaTT`. Dataset rights remain with the original providers. See [NOTICE](NOTICE) for details.

The upstream MPT-Rec repository does not declare a license. The MIT license in this repository covers the original code authored here and does not relicense upstream-derived files.

## Citation

```bibtex
@article{tcka2026,
  title   = {基于任务条件知识增强的可扩展多任务推荐方法},
  author  = {李冯孺},
  journal = {软件学报},
  year    = {2026},
  note    = {Manuscript under review}
}
```
