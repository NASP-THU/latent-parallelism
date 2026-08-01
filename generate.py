# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# UNITE (Latent Parallelism) artifact entrypoint for NSDI '27 AE.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

# Force NCCL over PCIe even when NVLink is present (paper/AE default).
# Set ALLOW_NVLINK=1 to keep vendor NVLink/P2P behavior for local debugging.
if os.environ.get("ALLOW_NVLINK", "0") != "1":
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-14B": {
        "prompt":
            "A timelapse of a puddle drying up and disappearing on a hot day",
    },
}

AE_TASKS = ("t2v-14B",)


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify --ckpt_dir (or set CKPT_DIR)."
    assert args.task in AE_TASKS, (
        f"Unsupported task for this artifact: {args.task}. "
        f"Supported: {', '.join(AE_TASKS)}"
    )
    assert args.task in WAN_CONFIGS, f"Unknown task config: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 50
    if args.sample_shift is None:
        args.sample_shift = 5.0
    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for task {args.task}; "
        f"supported: {', '.join(SUPPORTED_SIZES[args.task])}"
    )

    if args.latent_parallel and args.lp_overlap_ratio is None:
        args.lp_overlap_ratio = 0.4


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UNITE: Communication-Efficient Distributed Inference for "
            "Video Diffusion Models with Latent Parallelism"
        ))
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(AE_TASKS),
        help="Text-to-video task / model size.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="Output resolution as width*height.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="Number of frames to generate (must be 4n+1).")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.environ.get("CKPT_DIR"),
        help="Checkpoint directory (or set env CKPT_DIR).")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Offload DiT to CPU between steps to save VRAM.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses sequence-parallel degree.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="Ring Attention sequence-parallel degree.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Shard T5 with FSDP.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Keep T5 on CPU (no t5_fsdp).")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Shard DiT with FSDP.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Output video path (.mp4).")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="RNG seed (<0 => random).")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="Diffusion sampler.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="Denoising steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Flow-matching shift.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor Parallel world size (must equal nproc when >1).")
    parser.add_argument(
        "--latent_parallel",
        action="store_true",
        default=False,
        help=(
            "Enable UNITE Latent Parallelism (paper method). "
            "Partitions the latent tensor across GPUs with patch-aligned "
            "overlapping slices, dynamic partition-dimension scheduling, "
            "and optional first-step full-latent SP warmup. "
            "Mutually exclusive with SP / TP / dit_fsdp."
        ))
    parser.add_argument(
        "--lp_overlap_ratio",
        type=float,
        default=None,
        help=(
            "Max overlap ratio r_max for Latent Parallelism "
            "(paper default 0.4). Required when --latent_parallel is set; "
            "if omitted with --latent_parallel, defaults to 0.4."
        ))
    parser.add_argument(
        "--run_vbench",
        action="store_true",
        default=False,
        help="Optional: run VBench after generation (requires vbench install).")
    parser.add_argument(
        "--vbench_output_dir",
        type=str,
        default="./vbench_results",
        help="Directory for VBench outputs.")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _get_parallel_strategy_prefix(args):
    use_sp = args.ulysses_size > 1 or args.ring_size > 1
    use_tp = args.tensor_parallel_size > 1

    if args.latent_parallel:
        return "unite"
    if use_tp:
        return "tp"
    if use_sp:
        if args.ulysses_size > 1 and args.ring_size > 1:
            return "sp_ulysses_ring"
        if args.ulysses_size > 1:
            return "sp_ulysses"
        return "sp_ring"
    if args.dit_fsdp:
        return "fsdp"
    return "single"


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

    use_sp = args.ulysses_size > 1 or args.ring_size > 1
    use_tp = args.tensor_parallel_size > 1
    use_lp = args.latent_parallel

    if use_lp:
        if args.lp_overlap_ratio is None:
            raise ValueError(
                "--lp_overlap_ratio is required when --latent_parallel is set.")
        if use_sp:
            raise ValueError(
                "UNITE (--latent_parallel) and Sequence Parallel "
                "(--ulysses_size / --ring_size) are mutually exclusive.")
        if use_tp:
            raise ValueError(
                "UNITE (--latent_parallel) and Tensor Parallel "
                "(--tensor_parallel_size) are mutually exclusive.")
        if args.dit_fsdp:
            raise ValueError(
                "UNITE (--latent_parallel) and --dit_fsdp are mutually exclusive.")

    if use_tp and use_sp:
        raise ValueError(
            "Tensor Parallel and Sequence Parallel are mutually exclusive.")
    if use_tp and args.dit_fsdp:
        raise ValueError(
            "Tensor Parallel and --dit_fsdp are mutually exclusive.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), (
            "t5_fsdp and dit_fsdp require a distributed launch.")
        assert not use_sp, "Sequence Parallel requires a distributed launch."
        assert not use_tp, "Tensor Parallel requires a distributed launch."
        assert not use_lp, "UNITE Latent Parallelism requires a distributed launch."

    if use_tp:
        assert args.tensor_parallel_size == world_size, (
            f"tensor_parallel_size ({args.tensor_parallel_size}) must equal "
            f"world_size ({world_size}).")

    if use_sp:
        assert args.ulysses_size * args.ring_size == world_size, (
            "ulysses_size * ring_size must equal world_size.")
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"{cfg.num_heads=} not divisible by {args.ulysses_size=}.")

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")
    logging.info(
        "NCCL interconnect: NCCL_NVLS_ENABLE=%s (ALLOW_NVLINK=%s); "
        "AE default disables NVLink so traffic uses PCIe.",
        os.environ.get("NCCL_NVLS_ENABLE", "<unset>"),
        os.environ.get("ALLOW_NVLINK", "0"),
    )

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    logging.info(f"Input prompt: {args.prompt}")

    logging.info("Creating WanT2V pipeline.")
    try:
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=use_sp,
            t5_cpu=args.t5_cpu,
            use_tensor_parallel=use_tp,
            tensor_parallel_size=args.tensor_parallel_size,
            use_latent_parallel=use_lp,
            lp_overlap_ratio=args.lp_overlap_ratio if use_lp else 0.0,
        )

        logging.info("Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        if rank == 0:
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                         "_")[:50]
                strategy_prefix = _get_parallel_strategy_prefix(args)
                size_tag = args.size.replace("*", "x") if sys.platform == "win32" else args.size
                args.save_file = (
                    f"{strategy_prefix}_{args.task}_{size_tag}_"
                    f"{formatted_prompt}_{formatted_time}.mp4"
                )

            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

            if args.run_vbench and args.save_file is not None:
                logging.info("Starting VBench evaluation ...")
                import importlib.util
                vbench_path = os.path.join(
                    os.path.dirname(__file__),
                    "scripts", "optional", "evaluate_vbench.py")
                spec = importlib.util.spec_from_file_location(
                    "evaluate_vbench", vbench_path)
                mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                mod.run_vbench_evaluation(
                    video_path=args.save_file,
                    output_dir=args.vbench_output_dir,
                )

        logging.info("Finished.")
    finally:
        # Avoid "process group has NOT been destroyed" warnings and reduce the
        # chance of NCCL hangs when launching back-to-back multi-GPU jobs.
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception as e:
                logging.warning("destroy_process_group failed: %s", e)


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
