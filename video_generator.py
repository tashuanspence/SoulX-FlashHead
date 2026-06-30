import asyncio
import os
import time
import subprocess
import numpy as np
import librosa
import imageio
from collections import deque
from loguru import logger

from flash_head.inference import (
    get_infer_params,
    get_audio_embedding,
    prepare_session_base_data,
    run_pipeline_for_session,
    SessionContext,
)
from image_compositor import prepare_background, should_composite, composite_frame, scale_frame


async def generate_mp4(
    pipeline,
    driving_image_path: str,
    audio_path: str,
    output_path: str,
    base_seed: int = 42,
    use_face_crop: bool = False,
    silence_padding_sec: float = 0.0,
    preserve_aspect_ratio: bool = False,
    output_scale: int = 1,
    encoder_crf: int = 20,
    encoder_preset: str = "veryfast",
    request_id: str = None,
    expression_scale: float = 1.0,
) -> str:
    """
    Run full inference and return a muxed MP4 file path.
    Uses session-aware locking to prevent avatar cross-contamination.
    """
    from app import _session_manager

    start_wall = time.time()
    session_id = request_id or f"vid_{int(time.time() * 1000)}"

    # Prepare background for aspect ratio preservation if needed
    background_data = None
    if should_composite(driving_image_path, preserve_aspect_ratio):
        background_data = prepare_background(driving_image_path, model_output_size=512 * output_scale)
        if background_data:
            logger.info(
                f"[{session_id}] Aspect ratio preservation enabled: "
                f"output will be {background_data[3]}x{background_data[4]}"
            )

    # Create session context for this video generation
    ctx = SessionContext(
        session_id=session_id,
        cond_image_path_or_dir=driving_image_path,
        base_seed=base_seed,
        use_face_crop=use_face_crop,
        expression_scale=expression_scale,
    )

    # IMPORTANT: use gpu_semaphore (not _setup_lock) so prepare_params() cannot
    # run concurrently with generate() on the shared pipeline object.
    loop = asyncio.get_event_loop()
    async with _session_manager.gpu_semaphore:
        await loop.run_in_executor(
            None,
            prepare_session_base_data,
            pipeline,
            ctx,
        )

    infer_params = get_infer_params()
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    # Load audio
    human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Prepend silence if requested
    if silence_padding_sec > 0:
        silence_samples = int(silence_padding_sec * sample_rate)
        silence = np.zeros(silence_samples, dtype=human_speech_array_all.dtype)
        human_speech_array_all = np.concatenate([silence, human_speech_array_all])
        logger.info(
            f"Prepended {silence_padding_sec}s silence ({silence_samples} samples)"
        )

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    audio_dq = deque(
        [0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum
    )

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

    num_full_chunks = len(human_speech_array_all) // human_speech_array_slice_len
    full_chunks_length = num_full_chunks * human_speech_array_slice_len

    slices = human_speech_array_all[:full_chunks_length].reshape(
        -1, human_speech_array_slice_len
    )
    remainder = human_speech_array_all[full_chunks_length:]

    logger.info(
        f"[video_generator] chunks={len(slices)}, remainder={len(remainder)} samples"
    )

    generated_list = []

    for chunk_idx, chunk_audio in enumerate(slices):
        audio_dq.extend(chunk_audio.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(
            pipeline, audio_array, audio_start_idx, audio_end_idx
        )

        # Run inference with semaphore protection to prevent avatar cross-contamination
        async with _session_manager.gpu_semaphore:
            video_tensor = await loop.run_in_executor(
                None, run_pipeline_for_session, pipeline, ctx, audio_embedding
            )
        frames = video_tensor.cpu()

        start_idx = motion_frames_num if chunk_idx == 0 else (frames.shape[0] - slice_len)
        generated_list.append(frames[start_idx:])

    # Final partial chunk
    if len(remainder) > 0:
        logger.info(
            f"[{session_id}] Processing final partial chunk ({len(remainder)} samples)"
        )
        padded = np.pad(
            remainder,
            (0, human_speech_array_slice_len - len(remainder)),
            mode="constant",
        )
        audio_dq.extend(padded.tolist())
        audio_array = np.array(audio_dq)

        remainder_frames = int(np.ceil(len(remainder) / sample_rate * tgt_fps))
        padding_frames = slice_len - remainder_frames

        dynamic_audio_end_idx = audio_end_idx - padding_frames
        dynamic_audio_start_idx = dynamic_audio_end_idx - frame_num

        audio_embedding = get_audio_embedding(
            pipeline, audio_array, dynamic_audio_start_idx, dynamic_audio_end_idx
        )
        # Run inference with semaphore protection
        async with _session_manager.gpu_semaphore:
            video_tensor = await loop.run_in_executor(
                None, run_pipeline_for_session, pipeline, ctx, audio_embedding
            )
        frames = video_tensor.cpu()
        generated_list.append(frames[:remainder_frames])

    # Write frames to temp video, then mux with audio
    if background_data:
        bg_array, x_offset, y_offset, out_w, out_h = background_data
        _save_video(
            generated_list, output_path, audio_path, tgt_fps,
            background=bg_array, x_offset=x_offset, y_offset=y_offset,
            audio_start_offset=silence_padding_sec,
            output_scale=output_scale,
            encoder_crf=encoder_crf,
            encoder_preset=encoder_preset,
        )
    else:
        _save_video(generated_list, output_path, audio_path, tgt_fps,
                    audio_start_offset=silence_padding_sec,
                    output_scale=output_scale,
                    encoder_crf=encoder_crf,
                    encoder_preset=encoder_preset)

    elapsed = time.time() - start_wall
    logger.info(f"[{session_id}] MP4 saved to {output_path} in {elapsed:.2f}s")
    return output_path


def _save_video(
    frames_list,
    video_path: str,
    audio_path: str,
    fps: int,
    background: np.ndarray = None,
    x_offset: int = 0,
    y_offset: int = 0,
    audio_start_offset: float = 0.0,
    output_scale: int = 1,
    encoder_crf: int = 20,
    encoder_preset: str = "veryfast",
):
    """Write frames to a temp video file, then mux with original audio via ffmpeg."""
    temp_video_path = video_path.replace(".mp4", "_tmp.mp4")

    # Pre-allocate output buffer if compositing
    output_buffer = np.empty_like(background) if background is not None else None

    with imageio.get_writer(
        temp_video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0", "-crf", str(encoder_crf), "-preset", encoder_preset],
    ) as writer:
        for frames in frames_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                frame = scale_frame(frames_np[i], output_scale)
                if background is not None:
                    # Composite frame onto background
                    frame = composite_frame(
                        frame, background, x_offset, y_offset, output_buffer
                    )
                writer.append_data(frame)

    cmd = ["ffmpeg", "-y", "-i", temp_video_path]
    if audio_start_offset > 0.0:
        cmd += ["-itsoffset", f"{audio_start_offset:.6f}"]
    cmd += [
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-af", "aresample=async=1:first_pts=0",
        "-shortest",
        video_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
