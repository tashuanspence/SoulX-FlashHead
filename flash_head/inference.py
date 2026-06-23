# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import yaml
import torch
import copy
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
from flash_head.src.distributed.usp_device import get_device, get_parallel_degree

with open("flash_head/configs/infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)


@dataclass
class SessionContext:
    """
    Holds the per-session driving-image conditioning state produced by
    pipeline.prepare_params().  Each concurrent stream owns one of these,
    preventing sessions from overwriting each other's state on the shared
    pipeline object.
    """
    session_id: str
    cond_image_path_or_dir: str
    base_seed: int
    use_face_crop: bool
    expression_scale: float = 1.0
    prepared: bool = False
    # Snapshot of all session-specific pipeline attributes captured after
    # prepare_params().  Restored before every generate() call so this session's
    # avatar conditioning is never contaminated by another session.
    # latent_motion_frames inside the snapshot is updated after each generate()
    # so cross-chunk temporal continuity is preserved automatically.
    _snapshot: dict = field(default_factory=dict)


def get_pipeline(world_size, ckpt_dir, model_type, wav2vec_dir):
    global infer_params
    ulysses_degree, ring_degree = get_parallel_degree(world_size, infer_params['num_heads'])
    device = get_device(ulysses_degree, ring_degree)
    logger.info(f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, device: {device}")

    pipeline = FlashHeadPipeline(
        checkpoint_dir=ckpt_dir,
        model_type=model_type,
        wav2vec_dir=wav2vec_dir,
        device=device,
        use_usp=(world_size > 1),
    )

    # compute motion_frames_num
    motion_frames_latent_num = infer_params['motion_frames_latent_num']
    motion_frames_num = (motion_frames_latent_num - 1) * pipeline.config.vae_stride[0] + 1
    infer_params['motion_frames_num'] = motion_frames_num

    # TODO: move to args
    if model_type == "pretrained":
        infer_params['sample_steps'] = 20
    else:
        infer_params['sample_steps'] = 4
    return pipeline


# ---------------------------------------------------------------------------
# Exact set of attributes written by FlashHeadPipeline.prepare_params() and
# reset_person_name(), sourced from the upstream pipeline implementation at
# https://github.com/Soul-AILab/SoulX-FlashHead/blob/main/flash_head/src/pipeline/flash_head_pipeline.py
#
# generate() additionally mutates `latent_motion_frames` (the temporal motion
# context used for cross-chunk continuity) — this is snapshotted and restored
# automatically so each session maintains its own temporal state.
# ---------------------------------------------------------------------------
_PIPELINE_SESSION_ATTRS = [
    # set by prepare_params()
    "cond_image_dict",           # dict[str, PIL.Image]
    "frame_num",                 # int
    "motion_frames_num",         # int
    "color_correction_strength", # float
    "target_h",                  # int
    "target_w",                  # int
    "lat_h",                     # int
    "lat_w",                     # int
    "generator",                 # torch.Generator (RNG state)
    "timesteps",                 # list[Tensor]
    "cond_image_tensor_dict",    # dict[str, Tensor]
    "ref_img_latent_dict",       # dict[str, Tensor]
    # set by reset_person_name() (called inside prepare_params)
    "person_name",               # str
    "original_color_reference",  # Tensor
    "ref_img_latent",            # Tensor
    "latent_motion_frames",      # Tensor — mutated by generate() each chunk
]


_pipeline_clean_states: dict = {}


def capture_pipeline_clean_state(pipeline) -> None:
    """
    Save a baseline snapshot of the pipeline BEFORE any prepare_params() has
    been called.  Call this exactly once, immediately after the pipeline is
    created, so every subsequent session setup can start from a neutral state.
    """
    key = id(pipeline)
    if key in _pipeline_clean_states:
        return
    snap = {}
    for attr in _PIPELINE_SESSION_ATTRS:
        if hasattr(pipeline, attr):
            val = getattr(pipeline, attr)
            if isinstance(val, torch.Tensor):
                snap[attr] = val.clone()
            elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                snap[attr] = [t.clone() for t in val]
            else:
                snap[attr] = copy.deepcopy(val) if val is not None else val
    _pipeline_clean_states[key] = snap
    logger.info(
        f"[inference] Clean pipeline state captured for pipeline id={key} "
        f"({len(snap)} attributes)"
    )


def _restore_pipeline_clean_state(pipeline) -> bool:
    """
    Restore the pipeline to its pre-prepare_params baseline.

    If the captured clean state is empty (attributes don't exist on the pipeline
    at init time — they are only created by prepare_params()), we instead delete
    all known session attributes so the pipeline is in the same state as if
    prepare_params() had never been called.
    """
    snap = _pipeline_clean_states.get(id(pipeline))
    if snap is None:
        logger.warning(
            f"[inference] No clean state available for pipeline id={id(pipeline)}; "
            "skipping reset — first call or capture_pipeline_clean_state() not called yet"
        )
        return False

    if snap:
        # Non-empty snapshot: restore saved values
        _restore_pipeline(pipeline, snap)
        logger.debug(f"[inference] Restored {len(snap)} attributes from clean state")
    else:
        # Empty snapshot means none of the session attrs existed at pipeline init.
        # Delete them all now so prepare_params() starts from a truly blank slate,
        # not one contaminated by a previous session's generate() calls.
        deleted = []
        for attr in _PIPELINE_SESSION_ATTRS:
            if hasattr(pipeline, attr):
                try:
                    delattr(pipeline, attr)
                    deleted.append(attr)
                except AttributeError:
                    setattr(pipeline, attr, None)
                    deleted.append(f"{attr}=None")
        logger.debug(
            f"[inference] Clean state was empty — deleted session attrs to reset pipeline: {deleted}"
        )
    return True


def _snapshot_pipeline(pipeline) -> dict:
    """Capture all session-specific pipeline attributes into a plain dict."""
    snap = {}
    for attr in _PIPELINE_SESSION_ATTRS:
        if not hasattr(pipeline, attr):
            continue
        val = getattr(pipeline, attr)
        if isinstance(val, torch.Tensor):
            snap[attr] = val.clone()
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            snap[attr] = [t.clone() for t in val]
        elif isinstance(val, dict):
            # dict of tensors (cond_image_tensor_dict, ref_img_latent_dict)
            # or dict of PIL Images (cond_image_dict) — images are read-only so
            # shallow copy is safe; tensors are cloned to prevent aliasing.
            cloned = {}
            for k, v in val.items():
                cloned[k] = v.clone() if isinstance(v, torch.Tensor) else v
            snap[attr] = cloned
        elif isinstance(val, torch.Generator):
            # Preserve RNG state so each session's noise sequence is reproducible
            snap[attr] = ('__generator_state__', val.get_state())
        else:
            snap[attr] = val  # int, float, str — safe to reference directly
    return snap


def _restore_pipeline(pipeline, snapshot: dict):
    """Write a previously captured snapshot back onto the pipeline object."""
    for attr, val in snapshot.items():
        if isinstance(val, tuple) and len(val) == 2 and val[0] == '__generator_state__':
            # Restore torch.Generator state
            if hasattr(pipeline, attr) and isinstance(getattr(pipeline, attr), torch.Generator):
                getattr(pipeline, attr).set_state(val[1])
            else:
                gen = torch.Generator(device=pipeline.device)
                gen.set_state(val[1])
                setattr(pipeline, attr, gen)
        else:
            setattr(pipeline, attr, val)


def prepare_session_base_data(
    pipeline,
    ctx: SessionContext,
) -> None:
    """
    Run pipeline.prepare_params() for this session and store the resulting
    state in ctx._snapshot.  Idempotent: subsequent calls with the same ctx
    are no-ops (re-preparing only if the driving image / seed changed).
    """
    if ctx.prepared:
        logger.debug(f"[{ctx.session_id}] base_data already prepared, skipping")
        return

    from app import log_event, traced_span

    log_event(
        "session_prepare_base_data_started",
        session_id=ctx.session_id,
        base_seed=ctx.base_seed,
        use_face_crop=ctx.use_face_crop,
    )

    # Reset pipeline to clean baseline BEFORE prepare_params() so that temporal
    # state left by a previous session's generate() calls cannot contaminate this
    # session's conditioning (the root cause of avatar cross-contamination).
    reset_ok = _restore_pipeline_clean_state(pipeline)
    logger.debug(
        f"[{ctx.session_id}] Pipeline reset to clean state before prepare_params: {reset_ok}"
    )

    with traced_span(
        "session.prepare_base_data",
        session_id=ctx.session_id,
        base_seed=ctx.base_seed,
        use_face_crop=ctx.use_face_crop,
    ):
        pipeline.prepare_params(
            cond_image_path_or_dir=ctx.cond_image_path_or_dir,
            target_size=(infer_params['height'], infer_params['width']),
            frame_num=infer_params['frame_num'],
            motion_frames_num=infer_params['motion_frames_num'],
            sampling_steps=infer_params['sample_steps'],
            seed=ctx.base_seed,
            shift=infer_params['sample_shift'],
            color_correction_strength=infer_params['color_correction_strength'],
            use_face_crop=ctx.use_face_crop,
        )

    ctx._snapshot = _snapshot_pipeline(pipeline)
    ctx.prepared = True
    logger.info(f"[{ctx.session_id}] base_data prepared and snapshot saved")
    log_event("session_prepare_base_data_completed", session_id=ctx.session_id, status="success")


def run_pipeline_for_session(pipeline, ctx: SessionContext, audio_embedding):
    """
    Restore this session's pipeline state, run inference, then save the updated
    state back.  Must be called with the GPU semaphore already acquired.

    The snapshot captures the full set of session-specific attributes (sourced
    from the upstream FlashHeadPipeline implementation), including
    latent_motion_frames which generate() updates after each chunk.  Restoring
    the snapshot before generate() therefore gives both correct avatar
    conditioning AND correct temporal continuity — no per-chunk prepare_params()
    overhead required.
    """
    if not ctx.prepared:
        raise RuntimeError(
            f"[{ctx.session_id}] prepare_session_base_data() must be called before run_pipeline_for_session()"
        )
    from app import traced_span

    with traced_span("generation.chunk_inference", session_id=ctx.session_id):
        # Restore this session's full pipeline state (avatar image, timesteps,
        # latent_motion_frames from last chunk, generator RNG state, etc.)
        _restore_pipeline(pipeline, ctx._snapshot)

        expression_scale = getattr(ctx, "expression_scale", 1.0)
        # Fast path: skip all expression handling when using the default value so
        # inference is exactly as fast as before the feature was added.
        if expression_scale != 1.0:
            logger.info(
                f"[{ctx.session_id}] Applying expression_scale={expression_scale} "
                f"to audio projection output via pipeline.generate"
            )
            result = run_pipeline(pipeline, audio_embedding, context_scale=expression_scale)
        else:
            result = run_pipeline(pipeline, audio_embedding)

        # Save updated state so next chunk for THIS session gets the correct
        # latent_motion_frames (the last generated frames for temporal continuity).
        ctx._snapshot = _snapshot_pipeline(pipeline)
        return result


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility with generate_video.py and
# the CLI generate_video.py script (single-session, non-concurrent paths).
# ---------------------------------------------------------------------------

# Module-level cache only for the single-session (non-concurrent) code paths.
_base_data_cache = {}


def get_base_data(pipeline, cond_image_path_or_dir, base_seed, use_face_crop):
    """Legacy single-session helper.  Use prepare_session_base_data for concurrent streams."""
    global _base_data_cache
    cache_key = (id(pipeline), cond_image_path_or_dir, base_seed, use_face_crop)

    if cache_key in _base_data_cache:
        logger.info(f"Using cached base_data for {cond_image_path_or_dir}")
        return

    pipeline.prepare_params(
        cond_image_path_or_dir=cond_image_path_or_dir,
        target_size=(infer_params['height'], infer_params['width']),
        frame_num=infer_params['frame_num'],
        motion_frames_num=infer_params['motion_frames_num'],
        sampling_steps=infer_params['sample_steps'],
        seed=base_seed,
        shift=infer_params['sample_shift'],
        color_correction_strength=infer_params['color_correction_strength'],
        use_face_crop=use_face_crop,
    )

    _base_data_cache[cache_key] = True


def get_infer_params():
    global infer_params
    return copy.deepcopy(infer_params)


def get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
    # audio_array = loudness_norm(audio_array, infer_params['sample_rate'])
    audio_embedding = pipeline.preprocess_audio(audio_array, sr=infer_params['sample_rate'], fps=infer_params['tgt_fps'])

    if audio_start_idx == -1 or audio_end_idx == -1:
        audio_start_idx = 0
        audio_end_idx = audio_embedding.shape[0]

    indices = (torch.arange(2 * 2 + 1) - 2) * 1

    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx-1)

    audio_embedding = audio_embedding[center_indices][None,...].contiguous()
    return audio_embedding


def run_pipeline(pipeline, audio_embedding, context_scale: float = 1.0):
    audio_embedding = audio_embedding.to(pipeline.device)
    if context_scale != 1.0:
        sample = pipeline.generate(audio_embedding, context_scale=context_scale)
    else:
        sample = pipeline.generate(audio_embedding)
    sample_frames = (((sample+1)/2).permute(1,2,3,0).clip(0,1) * 255).contiguous()
    return sample_frames

