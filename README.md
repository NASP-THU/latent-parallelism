# UNITE (NSDI '27 Artifact)

**UNITE**: *Communication-Efficient Distributed Inference for Video Diffusion Models with Latent Parallelism*

- **Paper (accepted PDF):** [`LP_NSDI27_Accepted.pdf`](./LP_NSDI27_Accepted.pdf)
- **Stable artifact URL:** https://github.com/NASP-THU/latent-parallelism
- **Please evaluate Release / tag `v0.1.0`:** https://github.com/NASP-THU/latent-parallelism/releases/tag/v0.1.0

This README follows the NSDI artifact-evaluation packaging guidelines: **Getting Started Instructions**, **Detailed Instructions**, and explicit **Artifact Claims**.

---

## Artifact Claims

The AEC should evaluate this artifact against the claims below (not against every number or experiment in the paper). Absolute latency and communication volumes vary by GPU SKU and interconnect; we claim **relative behavior** that supports the paper’s main systems conclusion for the default AE path.

### Claims we ask the AEC to validate

1. **Functional latent-parallel inference.** On multi-GPU hardware meeting the Requirements below, `scripts/getting_started.sh` runs UNITE (`--latent_parallel`) for Wan2.1-T2V-14B and produces a playable MP4.
2. **Main latency claim (default AE path).** On **4 GPUs**, comparing UNITE vs Tensor Parallel (TP) under the same prompt / resolution / frames / steps (`scripts/run_nsdi_reproduce.sh`):
   - both strategies complete and write videos + logs;
   - the primary metric is **`denoising_s`** in `ae_outputs/nsdi_reproduce/latency_table.csv` (not end-to-end wall-clock, which includes weight load);
   - **UNITE `denoising_s` is substantially lower than TP `denoising_s`** on PCIe-style multi-GPU (scripts default `NCCL_NVLS_ENABLE=0` to match the paper’s commodity-interconnect setting).
3. **Optional baselines.** The same driver can also run Ulysses / Ring / FSDP for comparison; these are **optional** and not required for the default AE claim.

### Expected difficulties (please read before running)

- **Hardware:** ≥2 GPUs for Getting Started; **4× NVIDIA GPUs with ≥48 GB VRAM** for the default Detailed path.
- **Dependencies:** CUDA 12.x, conda, PyTorch 2.5.1, and `flash-attn` (build can take a long time and is machine-specific).
- **Checkpoint:** Wan2.1-T2V-14B weights are large; download is **not** part of the 30-minute kick-the-tires budget.
- **Logs may appear idle** for several minutes during 14B weight load; that is normal.
- **Interconnect:** scripts disable NVLink by default (`NCCL_NVLS_ENABLE=0`) for AE comparability with PCIe settings. Set `ALLOW_NVLINK=1` only if you intentionally want NVLink.

### Success criteria (concrete)

| Phase | Success |
|---|---|
| Getting Started | `ae_outputs/getting_started/unite.mp4` exists; log contains denoising / finished messages |
| Detailed (default) | `ae_outputs/nsdi_reproduce/{unite,tp}/` each has a video + `generation.log`; `latency_table.csv` has both rows; `denoising_s(unite) < denoising_s(tp)` with a clear gap |

---

## Requirements

- Linux + CUDA 12.x + Python 3.10 (via conda)
- **Getting Started:** ≥2 NVIDIA GPUs, each with ≥48 GB VRAM (4 GPUs typical)
- **Detailed (default AE):** **4** NVIDIA GPUs, each with ≥48 GB VRAM
- PyTorch 2.5.1, `flash-attn`, `xfuser`
- Disk for Wan2.1-T2V-14B checkpoint (tens of GB)

---

## Installation (one-time; not counted in the 30-minute kick-the-tires window)

From the repository root (prefer the `v0.1.0` tree):

```bash
bash scripts/install.sh
conda activate unite
```

This creates/activates the `unite` conda env, installs PyTorch 2.5.1 (CUDA 12.1 wheels), project dependencies, `flash-attn`, and the editable `unite` package, then verifies `torch` / `flash_attn` / `xfuser` / `wan` imports.

Manual equivalent:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
PIP_NO_CACHE_DIR=1 python -m pip install flash-attn --no-build-isolation --no-cache-dir
python -m pip install -e . --no-deps
python -c "import torch, flash_attn, xfuser, wan; print('ok')"
```

Checkpoint (one-time):

```bash
bash scripts/download_model.sh
# or
export CKPT_DIR=/path/to/Wan2.1-T2V-14B
```

`environment.yml` only bootstraps Python 3.10; always run `scripts/install.sh` / `scripts/setup_env.sh` for the full AE environment.

---

## Getting Started Instructions

**Goal (kick-the-tires):** confirm the artifact runs a small but real UNITE job and produces a video.

**Time budget:** about **30 minutes of reviewer interaction time**, assuming:

1. `bash scripts/install.sh` already succeeded and `conda activate unite` works;
2. Wan2.1-T2V-14B is already available at `./Wan2.1-T2V-14B` or `$CKPT_DIR`;
3. ≥2 visible GPUs with sufficient VRAM.

Environment setup and weight download are **prep steps** and may take much longer than 30 minutes on a cold machine; they are intentionally separated from this section.

### Steps

```bash
conda activate unite
# optional: export CKPT_DIR=/path/to/Wan2.1-T2V-14B
bash scripts/getting_started.sh
# optional overrides: NPROC=4 OUT_DIR=... PROMPT='...'
```

What this runs (Wan2.1-T2V-14B):

| Setting | Value |
|---|---|
| Resolution / frames | `832*480` / **17** |
| Steps | 50 (UniPC), guide 5.0, shift 5.0, seed 42 |
| Method | `--latent_parallel --lp_overlap_ratio 0.4` |
| GPUs | `$NPROC` (default: all visible GPUs; need ≥2) |
| Output | `ae_outputs/getting_started/unite.mp4` + `generation.log` |

Wall-clock with weights already local is typically **~8–20 minutes** (dominated by multi-minute 14B load). Denoising alone is shorter. If the log pauses during load, wait.

### Kick-the-tires checks

1. Script exits 0.
2. File `ae_outputs/getting_started/unite.mp4` exists and is non-empty.
3. `generation.log` mentions UNITE / total denoising time / finished (exact strings may vary slightly).

If checkpoint is missing, the script fails fast with download instructions—that is an expected kick-the-tires failure mode to catch incomplete setup.

---

## Detailed Instructions

**Goal (full evaluation):** reproduce the default AE latency comparison that supports Claim 2 above.

### Default AE path (recommended): UNITE vs Tensor Parallel on 4 GPUs

```bash
conda activate unite
# optional: export CKPT_DIR=/path/to/Wan2.1-T2V-14B
bash scripts/run_nsdi_reproduce.sh
# overrides: NPROC=4 FRAME_NUM=37 SIZE='832*480' SAMPLE_STEPS=50
```

Default configuration:

| Setting | Value |
|---|---|
| Model | Wan2.1-T2V-14B |
| GPUs | **4** |
| Resolution / frames | `832*480` / **37** |
| Steps | 50 (UniPC), guide 5.0, shift 5.0, seed 42, `r_max=0.4` |
| Strategies | `unite`, `tp` |
| Interconnect | PCIe-oriented (`NCCL_NVLS_ENABLE=0` unless `ALLOW_NVLINK=1`) |

Approximate wall-clock with local weights: about **20–40 minutes** (two strategy runs, each with weight load).

### Outputs and how to judge

Under `ae_outputs/nsdi_reproduce/`:

- `unite/unite.mp4`, `unite/generation.log`
- `tp/tp.mp4`, `tp/generation.log`
- `latency_table.csv` / `latency_table.json` / `summary.txt`

**Primary metric:** `denoising_s` (lower is better).  
**Pass condition for Claim 2:** UNITE is substantially faster than TP on `denoising_s`. Do **not** require matching any absolute paper table entry.

The table is produced by `scripts/parse_latency.py` from the per-strategy logs.

### Optional: more baselines

```bash
STRATEGIES=unite,ulysses,ring,fsdp,tp bash scripts/run_nsdi_reproduce.sh
```

This is slower and **not required** for the default AE claim. A more configurable driver is `scripts/run_latency_compare.sh`.

### Manual commands (equivalent to the default path)

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
```

---

## Repository layout

```
LP_NSDI27_Accepted.pdf      # accepted NSDI paper PDF
generate.py                 # inference entry (Wan2.1-T2V-14B)
wan/                        # model + UNITE / SP / TP / FSDP
scripts/
  install.sh                # one-click environment setup
  setup_env.sh
  download_model.sh         # Wan2.1-T2V-14B checkpoint
  getting_started.sh        # kick-the-tires UNITE smoke test
  run_nsdi_reproduce.sh     # default AE: 4-GPU UNITE vs TP
  run_latency_compare.sh    # configurable strategy compare
  parse_latency.py
  optional/                 # quality helpers (not default AE)
requirements.txt
environment.yml
LICENSE.txt
```

---

## License

See [`LICENSE.txt`](./LICENSE.txt). This artifact builds on Wan2.1 (Alibaba Wan Team); see the license for upstream attribution.

## Contact for AE

For clarifications during artifact evaluation, please use the HotCRP artifact submission site so reviewer anonymity is preserved.
