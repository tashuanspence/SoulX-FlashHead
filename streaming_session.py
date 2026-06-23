import asyncio
import inspect
from collections import deque
from typing import Optional, Callable

import numpy as np
from loguru import logger


class StreamingAudioBuffer:
    """
    Accumulates PCM audio samples as they arrive via WebSocket and
    tracks when enough samples exist for next inference chunk.
    """

    def __init__(self, sample_rate: int, chunk_duration_samples: int):
        """
        Args:
            sample_rate: Audio sample rate (e.g., 16000)
            chunk_duration_samples: Number of samples per inference chunk
        """
        self.sample_rate = sample_rate
        self.chunk_duration_samples = chunk_duration_samples
        self.buffer = []
        self.total_samples_received = 0
        self.total_chunks_yielded = 0

    def append(self, audio_samples: np.ndarray):
        """Append new audio samples to the buffer."""
        self.buffer.extend(audio_samples.tolist())
        self.total_samples_received += len(audio_samples)
        logger.debug(f"Buffer: +{len(audio_samples)} samples, total={len(self.buffer)}")

    def has_next_chunk(self) -> bool:
        """Check if we have enough samples for the next inference chunk."""
        return len(self.buffer) >= self.chunk_duration_samples

    def get_next_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next chunk of audio for inference.
        Returns None if not enough samples available.
        """
        if not self.has_next_chunk():
            return None

        chunk = np.array(self.buffer[: self.chunk_duration_samples], dtype=np.float32)
        self.buffer = self.buffer[self.chunk_duration_samples :]
        self.total_chunks_yielded += 1

        logger.debug(f"Chunk yielded: {len(chunk)} samples, remaining={len(self.buffer)}")
        return chunk

    def flush_remaining(self) -> Optional[np.ndarray]:
        """
        Get any remaining samples as a final chunk (padded if necessary).
        Returns None if buffer is empty.
        """
        if len(self.buffer) == 0:
            return None

        remaining_samples = len(self.buffer)
        if remaining_samples < self.chunk_duration_samples:
            # Pad with zeros to match expected chunk size
            padding = self.chunk_duration_samples - remaining_samples
            chunk = np.array(self.buffer + [0.0] * padding, dtype=np.float32)
            logger.info(f"Flushing final chunk: {remaining_samples} samples + {padding} padding")
        else:
            chunk = np.array(self.buffer[: self.chunk_duration_samples], dtype=np.float32)
            logger.info(f"Flushing chunk: {len(chunk)} samples")

        self.buffer = []
        return chunk

    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        logger.debug("Buffer cleared")

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "buffered_samples": len(self.buffer),
            "buffered_ms": int(len(self.buffer) / self.sample_rate * 1000),
            "total_received": self.total_samples_received,
            "total_chunks": self.total_chunks_yielded,
        }


class StreamingSession:
    """
    Manages a single real-time streaming audio→video session.

    Concurrency contract
    --------------------
    Each session holds its own ``SessionContext`` (driving-image conditioning
    snapshot) so that multiple concurrent sessions sharing the same pipeline
    object do NOT overwrite each other's state.

    A shared ``asyncio.Semaphore`` (provided by ``ConcurrentSessionManager``)
    is acquired before every GPU inference call and released immediately after,
    serialising GPU access while still allowing up to N sessions to be active.
    """

    def __init__(
        self,
        session_id: str,
        pipeline,
        ctx,                        # inference.SessionContext
        infer_params: dict,
        gpu_semaphore: asyncio.Semaphore,
    ):
        self.session_id = session_id
        self.pipeline = pipeline
        self.ctx = ctx
        self.infer_params = infer_params
        self.gpu_semaphore = gpu_semaphore

        # Audio buffer setup
        sample_rate = infer_params["sample_rate"]
        tgt_fps = infer_params["tgt_fps"]
        frame_num = infer_params["frame_num"]
        motion_frames_num = infer_params["motion_frames_num"]
        slice_len = frame_num - motion_frames_num

        # Calculate samples per chunk
        chunk_duration_samples = slice_len * sample_rate // tgt_fps

        self.audio_buffer = StreamingAudioBuffer(sample_rate, chunk_duration_samples)

        # Audio context window (deque for sliding window)
        cached_audio_duration = infer_params["cached_audio_duration"]
        cached_audio_length_sum = sample_rate * cached_audio_duration
        self.audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        # Frame tracking
        self.audio_end_idx = cached_audio_duration * tgt_fps
        self.audio_start_idx = self.audio_end_idx - frame_num
        self.frame_index = 0
        self.is_first_chunk = True
        self._stopped = False

        logger.info(
            f"[{session_id}] StreamingSession initialized: chunk_size={chunk_duration_samples} samples"
        )
        try:
            from app import log_event
            log_event("generation_started", session_id=session_id, generation_type="websocket_stream", status="started")
        except Exception:
            pass

    def add_audio_chunk(self, audio_samples: np.ndarray):
        """Add incoming audio samples to the buffer."""
        self.audio_buffer.append(audio_samples)

    async def _run_inference(self, audio_embedding):
        """
        Acquire the GPU semaphore, restore this session's driving-image state,
        run inference, and release the semaphore.  Running in a thread executor
        keeps the asyncio event loop unblocked during the GPU call.
        """
        from flash_head.inference import run_pipeline_for_session

        loop = asyncio.get_event_loop()

        def _gpu_work():
            return run_pipeline_for_session(self.pipeline, self.ctx, audio_embedding)

        async with self.gpu_semaphore:
            logger.debug(f"[{self.session_id}] GPU semaphore acquired")
            video_tensor = await loop.run_in_executor(None, _gpu_work)
            logger.debug(f"[{self.session_id}] GPU semaphore released")

        return video_tensor

    async def generate_frames(self, on_frame: Callable):
        """
        Generate video frames as audio chunks become available.

        Args:
            on_frame: Async or sync callback(frame_tensor, frame_index) called for each frame
        """
        from flash_head.inference import get_audio_embedding

        slice_len = self.infer_params["frame_num"] - self.infer_params["motion_frames_num"]

        while not self._stopped:
            # Wait for next chunk to be available
            if not self.audio_buffer.has_next_chunk():
                await asyncio.sleep(0.01)  # Small delay to avoid busy waiting
                continue

            chunk = self.audio_buffer.get_next_chunk()
            if chunk is None:
                continue

            # Update audio context window
            self.audio_dq.extend(chunk.tolist())
            audio_array = np.array(self.audio_dq)

            # Get audio embedding (CPU-only, no semaphore needed)
            audio_embedding = get_audio_embedding(
                self.pipeline,
                audio_array,
                self.audio_start_idx,
                self.audio_end_idx,
            )

            # Run inference with semaphore protection
            video_tensor = await self._run_inference(audio_embedding)
            frames = video_tensor.cpu()

            if self.is_first_chunk:
                start_idx = self.infer_params["motion_frames_num"]
                self.is_first_chunk = False
            else:
                start_idx = frames.shape[0] - slice_len

            for i in range(start_idx, frames.shape[0]):
                frame_tensor = frames[i]
                frame_result = on_frame(frame_tensor, self.frame_index)
                if inspect.isawaitable(frame_result):
                    await frame_result
                self.frame_index += 1

            logger.debug(
                f"[{self.session_id}] Generated {frames.shape[0] - start_idx} frames, total={self.frame_index}"
            )

            try:
                from app import log_event
                log_event("generation_chunk_completed", session_id=self.session_id, frames_generated=self.frame_index, chunk_frames=frames.shape[0] - start_idx)
            except Exception:
                pass

    async def finalize(self, on_frame: Callable):
        """Process any remaining audio and generate final frames."""
        from flash_head.inference import get_audio_embedding

        remaining_chunk = self.audio_buffer.flush_remaining()
        if remaining_chunk is None:
            logger.info(f"[{self.session_id}] finalized, no remaining audio")
            try:
                from app import log_event
                log_event("generation_completed", session_id=self.session_id, generation_type="websocket_stream", frames_generated=self.frame_index, status="success")
            except Exception:
                pass
            return

        # Calculate how many actual frames to generate based on remaining audio
        remaining_samples = np.count_nonzero(remaining_chunk)
        sample_rate = self.infer_params["sample_rate"]
        tgt_fps = self.infer_params["tgt_fps"]
        frame_num = self.infer_params["frame_num"]

        remainder_frames = int(np.ceil(remaining_samples / sample_rate * tgt_fps))

        # Update audio context window
        self.audio_dq.extend(remaining_chunk.tolist())
        audio_array = np.array(self.audio_dq)

        # Adjust indices for partial chunk
        slice_len = frame_num - self.infer_params["motion_frames_num"]
        padding_frames = slice_len - remainder_frames
        dynamic_audio_end_idx = self.audio_end_idx - padding_frames
        dynamic_audio_start_idx = dynamic_audio_end_idx - frame_num

        # Get audio embedding
        audio_embedding = get_audio_embedding(
            self.pipeline,
            audio_array,
            dynamic_audio_start_idx,
            dynamic_audio_end_idx,
        )

        # Run inference with semaphore protection
        video_tensor = await self._run_inference(audio_embedding)
        frames = video_tensor.cpu()

        # Yield only the frames corresponding to actual audio
        num_frames_to_yield = min(remainder_frames, frames.shape[0])
        for i in range(num_frames_to_yield):
            frame_tensor = frames[i]
            frame_result = on_frame(frame_tensor, self.frame_index)
            if inspect.isawaitable(frame_result):
                await frame_result
            self.frame_index += 1

        logger.info(
            f"[{self.session_id}] finalized: {num_frames_to_yield} final frames, total={self.frame_index}"
        )
        try:
            from app import log_event
            log_event("generation_completed", session_id=self.session_id, generation_type="websocket_stream", frames_generated=self.frame_index, final_frames=num_frames_to_yield, status="success")
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Get session statistics."""
        buffer_stats = self.audio_buffer.get_stats()
        return {
            "session_id": self.session_id,
            "frames_generated": self.frame_index,
            **buffer_stats,
        }

    def stop(self):
        """Signal the frame generation loop to stop."""
        self._stopped = True
