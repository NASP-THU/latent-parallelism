# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# UNITE / Latent Parallelism extensions for NSDI artifact evaluation.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.tensor_parallel import setup_tensor_parallel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        use_tensor_parallel=False,
        tensor_parallel_size=1,
        use_latent_parallel=False,
        lp_overlap_ratio=0.5,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        logging.debug("init: loading T5...")
        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        logging.debug("init: T5 loaded, loading VAE...")
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.debug(f"init: VAE loaded, loading DiT from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir, torch_dtype=config.param_dtype)
        self.model.eval().requires_grad_(False)
        logging.debug(f"init: DiT loaded ({len(self.model.blocks)} blocks)")

        self.use_latent_parallel = use_latent_parallel
        self.lp_overlap_ratio = lp_overlap_ratio
        self._lp_opt_warmup_sp_size = 1
        self._lp_opt_usp_ready = False
        self._lp_opt_usp_active = False
        self._lp_opt_original_model_forward = None
        self._lp_opt_original_attn_forwards = None
        self._lp_opt_usp_model_forward = None
        self._lp_opt_usp_attn_forwards = None

        if use_latent_parallel:
            # UNITE / Latent Parallelism: full DiT replica per GPU; partition latents.
            self.sp_size = 1
            if dist.is_initialized():
                dist.barrier()
            self.model.to(self.device)
            self._lp_opt_prepare_sequence_parallel_warmup()
        elif use_tensor_parallel:
            # Tensor Parallel path: shard attention heads + FFN, add all-reduce
            self.sp_size = 1
            self.model = setup_tensor_parallel(
                model=self.model,
                tp_rank=rank,
                tp_size=tensor_parallel_size,
                device=self.device,
            )
        elif use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()

            if dist.is_initialized():
                dist.barrier()
            if dit_fsdp:
                self.model = shard_fn(self.model)
            else:
                self.model.to(self.device)
        else:
            self.sp_size = 1

            if dist.is_initialized():
                dist.barrier()
            if dit_fsdp:
                self.model = shard_fn(self.model)
            else:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def _lp_opt_prepare_sequence_parallel_warmup(self):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        from xfuser.core.distributed import (
            get_sequence_parallel_world_size,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from .distributed.xdit_context_parallel import (
            usp_attn_forward,
            usp_dit_forward,
        )

        logging.debug("unite: preparing sequence-parallel warmup")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=dist.get_world_size(),
            ulysses_degree=1,
        )

        self._lp_opt_original_model_forward = self.model.forward
        self._lp_opt_original_attn_forwards = [
            block.self_attn.forward for block in self.model.blocks
        ]
        self._lp_opt_usp_model_forward = types.MethodType(
            usp_dit_forward, self.model)
        self._lp_opt_usp_attn_forwards = [
            types.MethodType(usp_attn_forward, block.self_attn)
            for block in self.model.blocks
        ]
        self._lp_opt_warmup_sp_size = get_sequence_parallel_world_size()
        self._lp_opt_usp_ready = True
        logging.debug(f"unite: sequence-parallel warmup ready (sp_size={self._lp_opt_warmup_sp_size})")

    def _lp_opt_set_sequence_parallel_warmup_mode(self, enabled):
        if not self._lp_opt_usp_ready or self._lp_opt_usp_active == enabled:
            return
        if enabled:
            logging.debug("unite: enabling sequence-parallel warmup mode")
            for block, attn_forward in zip(
                    self.model.blocks, self._lp_opt_usp_attn_forwards):
                block.self_attn.forward = attn_forward
            self.model.forward = self._lp_opt_usp_model_forward
        else:
            logging.debug("unite: disabling sequence-parallel warmup mode")
            for block, attn_forward in zip(
                    self.model.blocks, self._lp_opt_original_attn_forwards):
                block.self_attn.forward = attn_forward
            self.model.forward = self._lp_opt_original_model_forward
        self._lp_opt_usp_active = enabled

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        logging.debug(f"generate: entered, size={size}, frame_num={frame_num}")
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        logging.debug("generate: encoding text with T5...")
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if self.use_latent_parallel:
                latents = self._latent_parallel_optimized_denoise(
                    latents, timesteps, sample_scheduler, seed_g,
                    guide_scale, arg_c, arg_null, offload_model)
            else:
                import time as _time
                torch.cuda.synchronize()
                _t_denoise_start = _time.perf_counter()
                _step_times = []

                for _step_i, t in enumerate(tqdm(timesteps)):
                    torch.cuda.synchronize()
                    _t_step_s = _time.perf_counter()

                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents = [temp_x0.squeeze(0)]

                    torch.cuda.synchronize()
                    _step_times.append(_time.perf_counter() - _t_step_s)

                torch.cuda.synchronize()
                _t_denoise_end = _time.perf_counter()
                if self.rank == 0:
                    _total = _t_denoise_end - _t_denoise_start
                    _avg = sum(_step_times) / len(_step_times) if _step_times else 0
                    _strategy = "USP" if self.sp_size > 1 else "FSDP/single"
                    logging.info(
                        f"\n{'='*60}\n"
                        f"Denoising Timing ({_strategy}, sp_size={self.sp_size}):\n"
                        f"  Total denoising time:  {_total:.3f}s\n"
                        f"  Num steps:             {len(_step_times)}\n"
                        f"  Avg step time:         {_avg*1000:.1f}ms\n"
                        f"{'='*60}")

            x0 = latents
            # Always offload DiT before VAE decode to avoid OOM at high res
            self.model.cpu()
            torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    # ─── Latent Parallelism Denoising Loop ──────────────────────────

    def _lp_choose_divide_dim(self, step_idx, latent_shape, num_slices):
        """
        [Opt 4] Choose split dimension, falling back to H or W if the
        natural rotating dimension can't produce enough slices for all GPUs.
        """
        # Natural rotation: dim 1 (T), 2 (H), 3 (W)
        candidate = (step_idx % 3) + 1
        dim_size = latent_shape[candidate]
        patch_size = self.patch_size[candidate - 1]
        num_patches = dim_size // patch_size
        if num_patches >= num_slices:
            return candidate
        # Fallback: try H (dim 2), then W (dim 3)
        for fallback in [2, 3]:
            if fallback == candidate:
                continue
            fb_patches = latent_shape[fallback] // self.patch_size[fallback - 1]
            if fb_patches >= num_slices:
                return fallback
        # Last resort: use whichever has most patches
        return candidate

    def _lp_compute_slices(self, dim_size, patch_size_for_dim, num_slices, overlap_ratio):
        """Compute slice indices with overlap for a given dimension."""
        if num_slices <= 1 or dim_size < num_slices:
            num_patches_total = dim_size // patch_size_for_dim
            return [(0, dim_size)], num_patches_total

        num_patches_total = dim_size // patch_size_for_dim
        if num_patches_total < num_slices:
            chunk_size_in_patches = 1
        else:
            chunk_size_in_patches = math.ceil(num_patches_total / num_slices)
        overlap_in_patches = int(chunk_size_in_patches * overlap_ratio)

        slice_indices_list = []
        for k in range(num_slices):
            start_patch = k * chunk_size_in_patches
            end_patch = start_patch + chunk_size_in_patches
            expanded_start_patch = max(0, start_patch - overlap_in_patches)
            expanded_end_patch = min(num_patches_total, end_patch + overlap_in_patches)
            expanded_start = expanded_start_patch * patch_size_for_dim
            expanded_end = expanded_end_patch * patch_size_for_dim
            if expanded_start < expanded_end:
                slice_indices_list.append((expanded_start, expanded_end))

        return slice_indices_list, num_patches_total

    def _lp_compute_slice_seq_len(self, slice_tensor):
        """Compute the correct seq_len for a latent slice based on its actual shape.

        This avoids the padding bug where passing the full-latent seq_len causes
        model.forward to zero-pad slice tokens (e.g. 3510) to the full length
        (e.g. 14040), making all linear layers do redundant computation.
        """
        # slice_tensor shape: [C, T, H, W]
        T_tok = slice_tensor.shape[1] // self.patch_size[0]
        H_tok = slice_tensor.shape[2] // self.patch_size[1]
        W_tok = slice_tensor.shape[3] // self.patch_size[2]
        return T_tok * H_tok * W_tok


    def _lp_opt_get_overlap_ratio(self, step_idx, total_steps, max_overlap):
        """
        Dynamic overlap schedule: linear decay from max_overlap to a minimum floor.
        Early steps (structure formation) get high overlap for quality.
        Late steps (detail refinement) get reduced but still meaningful overlap.

        Floor = 50% of max_overlap ensures at least 2 patches of overlap even
        in the last step (for chunk_size=8: int(8*0.25)=2). The previous 15%
        floor caused zero overlap in the last ~7 steps (int(8*0.075)=0), which
        created hard seam artifacts visible in structural details like limbs.
        """
        progress = step_idx / max(total_steps - 1, 1)  # 0.0 -> 1.0
        floor = 0.50 * max_overlap  # never go below this
        return max_overlap * (1.0 - progress) + floor * progress

    def _lp_opt_choose_divide_dim(self, step_idx, total_steps, latent_shape, num_slices):
        progress = step_idx / max(total_steps - 1, 1)
        h_patches = latent_shape[2] // self.patch_size[1]
        w_patches = latent_shape[3] // self.patch_size[2]

        if progress >= 0.92:
            if h_patches >= num_slices:
                return 2
            if w_patches >= num_slices:
                return 3

        if progress >= 0.80:
            candidate = 2 if (step_idx % 2 == 0) else 3
            candidate_patches = h_patches if candidate == 2 else w_patches
            if candidate_patches >= num_slices:
                return candidate
            fallback = 3 if candidate == 2 else 2
            fallback_patches = w_patches if fallback == 3 else h_patches
            if fallback_patches >= num_slices:
                return fallback

        candidate = self._lp_choose_divide_dim(step_idx, latent_shape, num_slices)
        if progress >= 0.70 and candidate == 1:
            if h_patches >= num_slices:
                return 2
            if w_patches >= num_slices:
                return 3
        return candidate

    # def _lp_opt_choose_spatial_dim(self, step_idx, latent_shape):
    #     """
    #     Choose split dimension: alternate between H (dim 2) and W (dim 3) only.
    #     No temporal splitting — preserves motion coherence across frames.
    #     """
    #     # Alternate H/W each step
    #     candidates = [2, 3]
    #     dim = candidates[step_idx % 2]
    #     return dim

    # def _lp_opt_compute_slices_no_overlap(self, dim_size, patch_size_for_dim, num_slices):
    #     """Compute non-overlapping slice indices for zero-overlap steps."""
    #     num_patches = dim_size // patch_size_for_dim
    def _lp_opt_cosine_blend_weight(self, ramp_len, device):
        """Cosine ramp: smoother than linear, avoids hard transitions."""
        if ramp_len <= 0:
            return torch.empty(0, device=device)
        t = torch.linspace(0, 1, ramp_len, device=device)
        return 0.5 * (1.0 - torch.cos(t * math.pi))

    def _lp_opt_get_guidance_rescale(self, step_idx, total_steps):
        progress = step_idx / max(total_steps - 1, 1)
        if progress <= 0.2:
            return 0.0
        return 0.35 * ((progress - 0.2) / 0.8)

    def _lp_opt_apply_guidance_rescale(self, guided, cond, rescale):
        if rescale <= 0:
            return guided
        reduce_dims = tuple(range(1, guided.ndim))
        guided_f = guided.float()
        cond_f = cond.float()
        guided_std = guided_f.std(dim=reduce_dims, keepdim=True)
        cond_std = cond_f.std(dim=reduce_dims, keepdim=True)
        rescaled = guided_f * (cond_std / guided_std.clamp(min=1e-6))
        mixed = rescale * rescaled + (1.0 - rescale) * guided_f
        return mixed.to(dtype=guided.dtype)

    def _lp_opt_get_guidance_smoothing_strength(self, step_idx, total_steps):
        progress = step_idx / max(total_steps - 1, 1)
        if progress <= 0.55:
            return 0.0
        return 0.10 * ((progress - 0.55) / 0.45)

    def _lp_opt_get_guide_scale_factor(self, step_idx, total_steps):
        progress = step_idx / max(total_steps - 1, 1)
        if progress <= 0.6:
            return 1.0
        return 1.0 - 0.10 * ((progress - 0.6) / 0.4)

    def _lp_opt_get_smoothing_strength(self, step_idx, total_steps):
        progress = step_idx / max(total_steps - 1, 1)
        if progress <= 0.85:
            return 0.0
        return 0.02 * ((progress - 0.85) / 0.15)

    def _lp_opt_spatial_smooth_noise(self, noise_pred, strength):
        if strength <= 0:
            return noise_pred
        orig_dtype = noise_pred.dtype
        t_dim = noise_pred.shape[1]
        c_dim = noise_pred.shape[0]
        h_dim = noise_pred.shape[2]
        w_dim = noise_pred.shape[3]
        x = noise_pred.float().permute(1, 0, 2, 3).contiguous().view(t_dim * c_dim, 1, h_dim, w_dim)
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        smooth = smooth.view(t_dim, c_dim, h_dim, w_dim).permute(1, 0, 2, 3)
        mixed = noise_pred.float() * (1.0 - strength) + smooth * strength
        return mixed.to(dtype=orig_dtype)

    def _lp_opt_compute_rope_offsets(self, slice_tensor, full_latent, divide_dim, start, batch_size=1):
        """Compute RoPE position offsets for a spatial slice.

        Returns a [batch_size, 3] tensor with (off_f, off_h, off_w) in token space.
        """
        # Convert pixel-space start position to token-space offset
        offsets = [0, 0, 0]  # [f, h, w]
        if divide_dim == 1:  # temporal (shouldn't happen in opt, but handle it)
            offsets[0] = start // self.patch_size[0]
        elif divide_dim == 2:  # H
            offsets[1] = start // self.patch_size[1]
        elif divide_dim == 3:  # W
            offsets[2] = start // self.patch_size[2]
        return torch.tensor([offsets] * batch_size, dtype=torch.long)

    def _latent_parallel_optimized_denoise(
        self, latents, timesteps, sample_scheduler, seed_g,
        guide_scale, arg_c, arg_null, offload_model,
    ):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        max_overlap = self.lp_overlap_ratio
        num_slices = world_size
        total_steps = len(timesteps)
        warmup_steps = 1 if total_steps > 1 else 0

        full_latent = latents[0]

        # Pre-build batched context (batch cond+uncond)
        context_c = arg_c['context']
        context_u = arg_null['context']
        batched_context = [context_c[0], context_u[0]]

        # Timing
        import time
        torch.cuda.synchronize()
        t_denoise_start = time.perf_counter()
        step_times = []
        step_model_times = []
        step_comm_times = []

        pbar = tqdm(enumerate(timesteps), total=total_steps,
                    desc="UNITE/LP Denoising", disable=(rank != 0))

        for i, t in pbar:
            torch.cuda.synchronize()
            t_step_start = time.perf_counter()

            if offload_model:
                self.model.to(self.device)

            if i == 0:
                dist.broadcast(full_latent, src=0)

            timestep_tensor = torch.tensor([t], device=self.device)
            guidance_rescale = self._lp_opt_get_guidance_rescale(i, total_steps)
            guidance_smoothing_strength = self._lp_opt_get_guidance_smoothing_strength(i, total_steps)
            guide_scale_factor = self._lp_opt_get_guide_scale_factor(i, total_steps)
            effective_guide_scale = guide_scale * guide_scale_factor
            smoothing_strength = self._lp_opt_get_smoothing_strength(i, total_steps)

            if i < warmup_steps:
                t_model_acc = 0.0
                if world_size > 1 and not self._lp_opt_usp_ready:
                    raise RuntimeError(
                        "UNITE full-latent warmup requires sequence-parallel warmup initialization")

                batched_input = [full_latent, full_latent]
                batched_t = timestep_tensor.repeat(2)
                full_seq_len = self._lp_compute_slice_seq_len(full_latent)
                logging.debug(f"unite: warmup step {i} using full latent with seq_len={full_seq_len}")

                try:
                    self._lp_opt_set_sequence_parallel_warmup_mode(True)
                    torch.cuda.synchronize()
                    t_model_start = time.perf_counter()
                    noise_preds = self.model(
                        batched_input, t=batched_t,
                        context=batched_context, seq_len=full_seq_len)
                    torch.cuda.synchronize()
                    t_model_acc = time.perf_counter() - t_model_start
                finally:
                    self._lp_opt_set_sequence_parallel_warmup_mode(False)

                noise_pred_cond = noise_preds[0]
                noise_pred_uncond = noise_preds[1]
                guidance_delta = noise_pred_cond - noise_pred_uncond
                guidance_delta = self._lp_opt_spatial_smooth_noise(
                    guidance_delta, guidance_smoothing_strength)
                guided = noise_pred_uncond + effective_guide_scale * guidance_delta
                local_noise_pred = self._lp_opt_apply_guidance_rescale(
                    guided, noise_pred_cond, guidance_rescale)
                del noise_preds, noise_pred_cond, noise_pred_uncond

                t_comm_start = time.perf_counter()
                t_comm_end = t_comm_start

                full_latent = sample_scheduler.step(
                    local_noise_pred.unsqueeze(0), t,
                    full_latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0].squeeze(0)

                del local_noise_pred

                if offload_model:
                    self.model.cpu()
                    torch.cuda.empty_cache()

                torch.cuda.synchronize()
                t_step_end = time.perf_counter()

                step_times.append(t_step_end - t_step_start)
                step_model_times.append(t_model_acc)
                step_comm_times.append(t_comm_end - t_comm_start)
                continue

            cur_overlap = self._lp_opt_get_overlap_ratio(i, total_steps, max_overlap)
            divide_dim = self._lp_opt_choose_divide_dim(i, total_steps, full_latent.shape, num_slices)

            dim_size = full_latent.shape[divide_dim]
            patch_size_for_dim = self.patch_size[divide_dim - 1]

            slice_indices_list, num_patches_total = self._lp_compute_slices(
                dim_size, patch_size_for_dim, num_slices, cur_overlap)
            num_actual_slices = len(slice_indices_list)

            if num_patches_total >= num_slices:
                chunk_size_for_weight = math.ceil(num_patches_total / num_slices)
            else:
                chunk_size_for_weight = 1

            # Pre-compute full weight map (all ranks identical → no reduce needed)
            full_weight = torch.zeros(
                full_latent.shape, device=self.device, dtype=torch.float32)
            for k in range(num_actual_slices):
                start, end = slice_indices_list[k]
                slicer = [slice(None)] * full_latent.ndim
                slicer[divide_dim] = slice(start, end)
                s_shape_dim = end - start

                w = torch.ones(
                    *[full_latent.shape[d] if d != divide_dim else s_shape_dim
                      for d in range(full_latent.ndim)],
                    device=self.device, dtype=torch.float32)

                ideal_start = k * chunk_size_for_weight * patch_size_for_dim
                ideal_end = ideal_start + chunk_size_for_weight * patch_size_for_dim
                ov_start_len = ideal_start - start
                ov_end_len = end - ideal_end
                ramp_view_shape = [1] * full_latent.ndim

                if ov_start_len > 0:
                    rs = [slice(None)] * full_latent.ndim
                    rs[divide_dim] = slice(0, ov_start_len)
                    ramp = self._lp_opt_cosine_blend_weight(
                        w[tuple(rs)].shape[divide_dim], self.device)
                    rv = ramp_view_shape.copy()
                    rv[divide_dim] = ramp.shape[0]
                    w[tuple(rs)] *= ramp.view(rv)

                if ov_end_len > 0:
                    rs = [slice(None)] * full_latent.ndim
                    rs[divide_dim] = slice(s_shape_dim - ov_end_len, s_shape_dim)
                    ramp = self._lp_opt_cosine_blend_weight(
                        w[tuple(rs)].shape[divide_dim], self.device)
                    rv = ramp_view_shape.copy()
                    rv[divide_dim] = ramp.shape[0]
                    w[tuple(rs)] *= ramp.flip(0).view(rv)

                full_weight[tuple(slicer)] += w

            full_weight.clamp_(min=1e-6)

            local_noise_pred = torch.zeros_like(full_latent)
            t_model_acc = 0.0

            for k in range(num_actual_slices):
                if k % world_size != rank:
                    continue

                start, end = slice_indices_list[k]
                slicer = [slice(None)] * full_latent.ndim
                slicer[divide_dim] = slice(start, end)
                slicer_tuple = tuple(slicer)
                s = full_latent[slicer_tuple]

                slice_seq_len = self._lp_compute_slice_seq_len(s)

                # Compute correct RoPE offsets for this slice's global position
                rope_offs = self._lp_opt_compute_rope_offsets(
                    s, full_latent, divide_dim, start, batch_size=2)

                batched_input = [s, s]
                batched_t = timestep_tensor.repeat(2)

                torch.cuda.synchronize()
                t_model_start = time.perf_counter()
                noise_preds = self.model(
                    batched_input, t=batched_t,
                    context=batched_context, seq_len=slice_seq_len,
                    rope_offsets=rope_offs)
                torch.cuda.synchronize()
                t_model_acc += time.perf_counter() - t_model_start

                noise_pred_cond = noise_preds[0]
                noise_pred_uncond = noise_preds[1]
                guidance_delta = noise_pred_cond - noise_pred_uncond
                guidance_delta = self._lp_opt_spatial_smooth_noise(
                    guidance_delta, guidance_smoothing_strength)
                guided = noise_pred_uncond + effective_guide_scale * guidance_delta
                guided = self._lp_opt_apply_guidance_rescale(
                    guided, noise_pred_cond, guidance_rescale)
                del noise_preds, noise_pred_cond, noise_pred_uncond

                # Compute per-slice weight for pre-normalized blending
                s_shape_dim = end - start
                w = torch.ones_like(s, dtype=torch.float32)
                ideal_start_k = k * chunk_size_for_weight * patch_size_for_dim
                ideal_end_k = ideal_start_k + chunk_size_for_weight * patch_size_for_dim
                ov_start_len = ideal_start_k - start
                ov_end_len = end - ideal_end_k

                if ov_start_len > 0:
                    rs = [slice(None)] * s.ndim
                    rs[divide_dim] = slice(0, ov_start_len)
                    ramp = self._lp_opt_cosine_blend_weight(
                        w[tuple(rs)].shape[divide_dim], self.device)
                    rv = [1] * s.ndim
                    rv[divide_dim] = ramp.shape[0]
                    w[tuple(rs)] *= ramp.view(rv)

                if ov_end_len > 0:
                    rs = [slice(None)] * s.ndim
                    rs[divide_dim] = slice(s_shape_dim - ov_end_len, s_shape_dim)
                    ramp = self._lp_opt_cosine_blend_weight(
                        w[tuple(rs)].shape[divide_dim], self.device)
                    rv = [1] * s.ndim
                    rv[divide_dim] = ramp.shape[0]
                    w[tuple(rs)] *= ramp.flip(0).view(rv)

                local_noise_pred[slicer_tuple] += (
                    guided * w / full_weight[slicer_tuple])

            # Single all_reduce
            torch.cuda.synchronize()
            t_comm_start = time.perf_counter()
            dist.all_reduce(local_noise_pred, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t_comm_end = time.perf_counter()

            del full_weight

            local_noise_pred = self._lp_opt_spatial_smooth_noise(
                local_noise_pred, smoothing_strength)

            # Scheduler step (all ranks identical)
            full_latent = sample_scheduler.step(
                local_noise_pred.unsqueeze(0), t,
                full_latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0].squeeze(0)

            del local_noise_pred

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            torch.cuda.synchronize()
            t_step_end = time.perf_counter()

            step_times.append(t_step_end - t_step_start)
            step_model_times.append(t_model_acc)
            step_comm_times.append(t_comm_end - t_comm_start)

        torch.cuda.synchronize()
        t_denoise_end = time.perf_counter()

        if rank == 0:
            total = t_denoise_end - t_denoise_start
            avg_step = sum(step_times) / len(step_times) if step_times else 0
            avg_model = sum(step_model_times) / len(step_model_times) if step_model_times else 0
            avg_comm = sum(step_comm_times) / len(step_comm_times) if step_comm_times else 0
            logging.info(
                f"\n{'='*60}\n"
                f"UNITE Latent Parallelism Denoising (max_overlap={max_overlap}):\n"
                f"  Total denoising time:    {total:.3f}s\n"
                f"  Num steps:               {len(step_times)}\n"
                f"  Avg step time:           {avg_step*1000:.1f}ms\n"
                f"  Avg model forward time:  {avg_model*1000:.1f}ms\n"
                f"  Avg comm (all_reduce):   {avg_comm*1000:.1f}ms\n"
                f"  Avg other overhead:      {(avg_step - avg_model - avg_comm)*1000:.1f}ms\n"
                f"{'='*60}")

        return [full_latent]

