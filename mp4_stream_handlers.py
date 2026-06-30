"""
MP4 stream handler functions for the /generate-mpeg-stream endpoint.
Kept in a separate module to avoid bloating app.py.
"""
import asyncio
import os
import uuid
from typing import AsyncGenerator, Optional
import time

import numpy as np
from loguru import logger

from config import FRAGMENT_DURATION_US
from metrics import distribution, generation_metric_tags, increment
from mp4_stream_encoder import Mp4StreamEncoder
from image_compositor import prepare_background, should_composite, scale_frame


async def generate_mp4_stream(
    driving_image_path: str,
    audio_path: str,
    base_seed: int,
    use_face_crop: bool,
    model_type: str,
    silence_padding_sec: float,
    output_scale: int = 1,
    encoder_crf: int = 20,
    encoder_preset: str = "veryfast",
    fragment_duration_us: int = FRAGMENT_DURATION_US,
    request_id: Optional[str] = None,
    preserve_aspect_ratio: bool = False,
    expression_scale: float = 1.0,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator that yields fragmented MP4 chunks.

    Mirrors generate_frames_stream() in app.py but routes frames through
    Mp4StreamEncoder instead of converting them to JPEG.  The caller
    wraps this in a FastAPI StreamingResponse with media_type='video/mp4'.
    """
    from observability import generation_log_fields, log_event, traced_span
    from app import (
        _session_manager,
        init_pipeline,
        normalize_model_type,
        FLASHHEAD_CKPT_DIR,
        WAV2VEC_DIR,
    )
    from flash_head.inference import (
        get_audio_embedding,
        get_infer_params,
        prepare_session_base_data,
        run_pipeline_for_session,
        SessionContext,
    )
    from session_manager import CapacityError
    import librosa
    from collections import deque

    try:
        model_type = normalize_model_type(model_type)
    except ValueError as exc:
        logger.error(str(exc))
        return

    tags = generation_metric_tags("mpeg_stream", model_type, endpoint="generate-mpeg-stream")

    session_id = request_id or f"mpeg_{uuid.uuid4().hex[:8]}"
    stream_start = time.time()

    init_start = time.time()
    pipeline = init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)
    logger.info(
        f"[{session_id}] pipeline init/lazy-load for model_type={model_type} "
        f"took {(time.time() - init_start) * 1000:.1f}ms"
    )

    ctx = SessionContext(
        session_id=session_id,
        cond_image_path_or_dir=driving_image_path,
        base_seed=base_seed,
        use_face_crop=use_face_crop,
        expression_scale=expression_scale,
    )

    # Prepare background for aspect ratio preservation if needed
    background_start = time.time()
    background_data = None
    if should_composite(driving_image_path, preserve_aspect_ratio):
        background_data = prepare_background(driving_image_path, model_output_size=512 * output_scale)
        if background_data:
            logger.info(
                f"[{session_id}] Aspect ratio preservation enabled: "
                f"output will be {background_data[3]}x{background_data[4]}"
            )
    logger.info(
        f"[{session_id}] background/aspect setup took "
        f"{(time.time() - background_start) * 1000:.1f}ms"
    )

    loop = asyncio.get_event_loop()
    # IMPORTANT: use gpu_semaphore (not _setup_lock) so prepare_params() and
    # generate() can NEVER run simultaneously on the shared pipeline object.
    # _setup_lock and gpu_semaphore are independent locks — holding one does
    # NOT block the other, which was the root cause of avatar cross-contamination.
    async with _session_manager.gpu_semaphore:
        logger.debug(f"[{session_id}] Acquiring gpu_semaphore for session setup (MPEG)")
        setup_start = time.time()
        await loop.run_in_executor(None, prepare_session_base_data, pipeline, ctx)
        logger.info(
            f"[{session_id}] prepare_session_base_data took "
            f"{(time.time() - setup_start) * 1000:.1f}ms"
        )
        logger.debug(f"[{session_id}] gpu_semaphore released after session setup (MPEG)")

    infer_params = get_infer_params()

    # Early capacity check to avoid race conditions
    if _session_manager.is_at_capacity():
        logger.warning(f"[{session_id}] Capacity full before session creation — rejecting MPEG stream")
        return

    session_created = False
    try:
        _session_manager.create_session(
            session_id=session_id,
            pipeline=pipeline,
            ctx=ctx,
            infer_params=infer_params,
        )
        session_created = True
    except CapacityError:
        logger.warning(f"[{session_id}] Capacity full — rejecting MPEG stream")
        return

    logger.info(
        f"[{session_id}] generation started: type=mpeg_stream model_type={model_type} "
        f"active={_session_manager.active_count()}/{_session_manager.max_streams}"
    )
    generation_start = time.time()
    log_event("generation_started", **generation_log_fields(session_id, "mpeg_stream", model_type, session_id=session_id, status="started"))

    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    audio_load_start = time.time()
    import soundfile as sf
    human_speech_array_all, sr_native = sf.read(audio_path, dtype="float32")
    if sr_native != sample_rate:
        import librosa
        human_speech_array_all = librosa.resample(human_speech_array_all, orig_sr=sr_native, target_sr=sample_rate)
    if len(human_speech_array_all.shape) > 1:
        human_speech_array_all = human_speech_array_all.mean(axis=1)
    logger.info(
        f"[{session_id}] audio load/resample took "
        f"{(time.time() - audio_load_start) * 1000:.1f}ms "
        f"sr_native={sr_native} target_sr={sample_rate} samples={len(human_speech_array_all)}"
    )

    if silence_padding_sec > 0:
        silence_samples = int(silence_padding_sec * sample_rate)
        silence = np.zeros(silence_samples, dtype=human_speech_array_all.dtype)
        human_speech_array_all = np.concatenate([silence, human_speech_array_all])

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

    num_full_chunks = len(human_speech_array_all) // human_speech_array_slice_len
    full_chunks_length = num_full_chunks * human_speech_array_slice_len
    human_speech_array_slices = human_speech_array_all[:full_chunks_length].reshape(
        -1, human_speech_array_slice_len
    )
    remainder = human_speech_array_all[full_chunks_length:]

    # Determine output dimensions based on aspect ratio preservation
    if background_data:
        bg_array, x_offset, y_offset, out_w, out_h = background_data
        encoder = Mp4StreamEncoder(
            width=out_w,
            height=out_h,
            fps=tgt_fps,
            audio_path=audio_path,
            job_id=session_id,
            crf=encoder_crf,
            preset=encoder_preset,
            fragment_duration_us=fragment_duration_us,
            background=bg_array,
            x_offset=x_offset,
            y_offset=y_offset,
            audio_start_offset=silence_padding_sec,
        )
    else:
        encoder = Mp4StreamEncoder(
            width=512 * output_scale,
            height=512 * output_scale,
            fps=tgt_fps,
            audio_path=audio_path,
            job_id=session_id,
            crf=encoder_crf,
            preset=encoder_preset,
            fragment_duration_us=fragment_duration_us,
            audio_start_offset=silence_padding_sec,
        )
    encoder_start_time = time.time()
    with traced_span("mpeg.encoder.start", session_id=session_id, fragment_duration_us=fragment_duration_us):
        encoder.start()
    logger.info(
        f"[{session_id}] encoder.start took "
        f"{(time.time() - encoder_start_time) * 1000:.1f}ms"
    )

    async def _drain_encoder(first_timeout: float = 0.01):
        """
        Drain all available chunks from the encoder queue.
        Blocks up to first_timeout for the first chunk to arrive,
        then drains the rest of the queue in a non-blocking manner.
        """
        import queue
        try:
            chunk = await loop.run_in_executor(None, encoder.get_chunk, first_timeout)
            if chunk is None:
                return
            yield chunk
        except Exception:
            return

        while True:
            try:
                chunk = encoder.chunk_queue.get_nowait()
                if chunk is None:
                    return
                yield chunk
            except queue.Empty:
                break

    try:
        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            chunk_start = time.time()
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            embedding_start = time.time()
            audio_embedding = get_audio_embedding(
                pipeline, audio_array, audio_start_idx, audio_end_idx
            )
            if chunk_idx == 0:
                logger.info(
                    f"[{session_id}] first chunk audio embedding took "
                    f"{(time.time() - embedding_start) * 1000:.1f}ms"
                )

            async with _session_manager.gpu_semaphore:
                video_tensor = await loop.run_in_executor(
                    None, run_pipeline_for_session, pipeline, ctx, audio_embedding
                )

            if chunk_idx == 0:
                logger.info(
                    f"[{session_id}] first chunk inference+embedding took "
                    f"{(time.time() - chunk_start) * 1000:.1f}ms"
                )

            cpu_copy_start = time.time()
            frames = video_tensor.cpu()
            if chunk_idx == 0:
                logger.info(
                    f"[{session_id}] first chunk cpu copy took "
                    f"{(time.time() - cpu_copy_start) * 1000:.1f}ms"
                )
            # Skip the first motion_frames_num warm-up frames on chunk 0.
            # Those frames are driven by zero audio context (silence) and would
            # shift the video lip-sync forward relative to the audio track,
            # causing perceived audio delay of motion_frames_num/fps seconds.
            # Subsequent chunks already drop them via (frames.shape[0] - slice_len).
            start_idx = motion_frames_num if chunk_idx == 0 else (frames.shape[0] - slice_len)
            add_frames_start = time.time()
            for i in range(start_idx, frames.shape[0]):
                frame_np = scale_frame(frames[i].numpy().astype(np.uint8), output_scale)
                encoder.add_frame(frame_np)
            if chunk_idx == 0:
                logger.info(
                    f"[{session_id}] first chunk frame enqueue took "
                    f"{(time.time() - add_frames_start) * 1000:.1f}ms frames={frames.shape[0] - start_idx}"
                )

            # Use a longer timeout (0.5s) on the very first chunk to guarantee
            # the initial frames are encoded and flushed immediately to the client
            # (maintaining ultra-low TTFF), while keeping a tiny non-blocking
            # timeout (0.01s) on subsequent chunks for high-throughput pipeline.
            timeout = 0.5 if chunk_idx == 0 else 0.01
            drain_start = time.time()
            first_yielded = False
            async for mp4_chunk in _drain_encoder(first_timeout=timeout):
                if chunk_idx == 0 and not first_yielded:
                    first_yielded = True
                    logger.info(
                        f"[{session_id}] first MP4 chunk ready after "
                        f"{(time.time() - stream_start) * 1000:.1f}ms "
                        f"drain_wait={(time.time() - drain_start) * 1000:.1f}ms"
                    )
                yield mp4_chunk

        # Process remainder
        if len(remainder) > 0:
            padded_remainder = np.pad(
                remainder,
                (0, human_speech_array_slice_len - len(remainder)),
                mode="constant",
            )

            audio_dq.extend(padded_remainder.tolist())
            audio_array = np.array(audio_dq)

            remainder_frames = int(np.ceil(len(remainder) / sample_rate * tgt_fps))
            padding_frames = slice_len - remainder_frames
            dynamic_audio_end_idx = audio_end_idx - padding_frames
            dynamic_audio_start_idx = dynamic_audio_end_idx - frame_num

            audio_embedding = get_audio_embedding(
                pipeline, audio_array, dynamic_audio_start_idx, dynamic_audio_end_idx
            )

            async with _session_manager.gpu_semaphore:
                video_tensor = await loop.run_in_executor(
                    None, run_pipeline_for_session, pipeline, ctx, audio_embedding
                )

            frames = video_tensor.cpu()
            for i in range(frames.shape[0] - remainder_frames, frames.shape[0]):
                frame_np = scale_frame(frames[i].numpy().astype(np.uint8), output_scale)
                encoder.add_frame(frame_np)

            async for mp4_chunk in _drain_encoder(first_timeout=0.01):
                yield mp4_chunk

        # Signal encoder to finish and drain remaining chunks
        with traced_span("mpeg.encoder.finish", session_id=session_id):
            encoder.finish()
        while True:
            chunk = encoder.get_chunk(timeout=0.5)
            if chunk is None:
                break
            if chunk:
                yield chunk

        logger.info(f"[{session_id}] generation completed: type=mpeg_stream")
        duration_ms = round((time.time() - generation_start) * 1000, 2)
        log_event("generation_completed", **generation_log_fields(session_id, "mpeg_stream", model_type, session_id=session_id, status="success", duration_ms=duration_ms))
        increment("soulx.generation.success", tags=generation_metric_tags("mpeg_stream", model_type, status="success", endpoint="generate-mpeg-stream"))
        distribution("soulx.generation.duration_ms", duration_ms, tags=tags)

    except GeneratorExit:
        # Client disconnected mid-stream
        logger.warning(f"[{session_id}] Client disconnected (GeneratorExit)")
        encoder.stop()
        raise
    except Exception as exc:
        logger.error(f"[{session_id}] MPEG stream error: {exc}")
        log_event("generation_failed", **generation_log_fields(session_id, "mpeg_stream", model_type, session_id=session_id, status="failure", error=str(exc)))
        increment("soulx.generation.failure", tags=generation_metric_tags("mpeg_stream", model_type, status="failure", endpoint="generate-mpeg-stream"))
        encoder.stop()
        raise

    finally:
        # Only cleanup if session was actually created
        if session_created:
            _session_manager.close_session(session_id)
            logger.info(f"[{session_id}] session closed: active={_session_manager.active_count()}/{_session_manager.max_streams}")