# UNITE (NSDI '27)

**UNITE**: *Communication-Efficient Distributed Inference for Video Diffusion Models with Latent Parallelism* 

---

## Quick start (4 GPUs)

```bash
bash scripts/install.sh
conda activate unite

bash scripts/download_model.sh          # or: export CKPT_DIR=/path/to/Wan2.1-T2V-14B
bash scripts/run_nsdi_reproduce.sh      # UNITE vs Tensor Parallel
```

Results are written to `ae_outputs/nsdi_reproduce/`, including per-strategy videos, logs, and `latency_table.csv`.

To also run Ulysses / Ring / FSDP:

```bash
STRATEGIES=unite,ulysses,ring,fsdp,tp bash scripts/run_nsdi_reproduce.sh
```

---

## Requirements

- 4× NVIDIA GPUs with ≥48 GB VRAM
- CUDA 12.x, Python 3.10
- PyTorch 2.5.1, flash-attn, xfuser

---

## Installation

```bash
bash scripts/install.sh
conda activate unite
```

This creates/activates a conda env, installs PyTorch 2.5.1 (CUDA 12.1), project dependencies, flash-attn, and the editable `unite` package, then verifies `torch` / `flash_attn` / `xfuser` / `wan` imports.

Manual equivalent:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
PIP_NO_CACHE_DIR=1 python -m pip install flash-attn --no-build-isolation --no-cache-dir
python -m pip install -e . --no-deps
python -c "import torch, flash_attn, xfuser, wan; print('ok')"
```

Checkpoint:

```bash
bash scripts/download_model.sh
# or
export CKPT_DIR=/path/to/Wan2.1-T2V-14B
```

---

## Getting Started

```bash
bash scripts/getting_started.sh
# optional: NPROC=4 OUT_DIR=... PROMPT='...'
```

---

## Results reproduction

```bash
bash scripts/run_nsdi_reproduce.sh
# overrides: NPROC=4 FRAME_NUM=37 SIZE='832*480' SAMPLE_STEPS=50
```

Default configuration:

| Setting | Value |
|---|---|
| Model | Wan2.1-T2V-14B |
| GPUs | 4 |
| Resolution / frames | `832*480` / 37 |
| Steps | 50 (UniPC) |
| Strategies | UNITE, Tensor Parallel |

Outputs under `ae_outputs/nsdi_reproduce/`:

- `latency_table.csv` / `latency_table.json` / `summary.txt`

Judge results by **`denoising_s`**: UNITE should be substantially faster than Tensor Parallel.

### Manual commands

```bash
N=4
CKPT_DIR=${CKPT_DIR:-./Wan2.1-T2V-14B}

# UNITE
python -m torch.distributed.run --nproc_per_node=$N generate.py \
  --task t2v-14B --size 832*480 --frame_num 37 \
  --ckpt_dir "$CKPT_DIR" --offload_model False --t5_fsdp \
  --sample_solver unipc --sample_steps 50 --sample_shift 5.0 \
  --sample_guide_scale 5.0 --base_seed 42 \
  --latent_parallel --lp_overlap_ratio 0.4 \
  --save_file out_unite.mp4

# Tensor Parallel
python -m torch.distributed.run --nproc_per_node=$N generate.py \
  --task t2v-14B --size 832*480 --frame_num 37 \
  --ckpt_dir "$CKPT_DIR" --offload_model False --t5_fsdp \
  --sample_solver unipc --sample_steps 50 --sample_shift 5.0 \
  --sample_guide_scale 5.0 --base_seed 42 \
  --tensor_parallel_size $N \
  --save_file out_tp.mp4

# Ulysses SP (optional baseline)
python -m torch.distributed.run --nproc_per_node=$N generate.py \
  --task t2v-14B --size 832*480 --frame_num 37 \
  --ckpt_dir "$CKPT_DIR" --offload_model False --t5_fsdp \
  --sample_solver unipc --sample_steps 50 --sample_shift 5.0 \
  --sample_guide_scale 5.0 --base_seed 42 \
  --ulysses_size $N --ring_size 1 \
  --save_file out_ulysses.mp4
```

---

## Repository layout

```
generate.py                 # inference entry (Wan2.1-T2V-14B)
wan/                        # model + UNITE / SP / TP / FSDP
scripts/
  install.sh                # one-click environment setup
  setup_env.sh
  download_model.sh         # Wan2.1-T2V-14B checkpoint
  getting_started.sh        # UNITE smoke test
  run_nsdi_reproduce.sh     # default AE: 4-GPU UNITE vs TP
  run_latency_compare.sh    # configurable strategy compare
  parse_latency.py
  optional/                 # VBench helpers
paper.pdf
requirements.txt
environment.yml
LICENSE.txt
```
