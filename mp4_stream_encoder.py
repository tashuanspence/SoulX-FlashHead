"""
Fragmented MP4 stream encoder for soulx-head.
Pipes raw RGB frames into FFmpeg, muxes audio, yields fMP4 chunks.
"""
import queue
import subprocess
import threading
from typing import Optional
import numpy as np
from loguru import logger
from config import SOULX_ENCODER_PRESET, SOULX_ENCODER_CRF
from image_compositor import composite_frame


class Mp4StreamEncoder:
    def __init__(self, width, height, fps, audio_path, job_id,
                 crf=None, preset=None, fragment_duration_us=200_000,
                 background: Optional[np.ndarray] = None,
                 x_offset: int = 0,
                 y_offset: int = 0,
                 audio_start_offset: float = 0.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_path = audio_path
        self.job_id = job_id
        self.crf = crf if crf is not None else SOULX_ENCODER_CRF
        self.preset = preset if preset is not None else SOULX_ENCODER_PRESET
        self.fragment_duration_us = fragment_duration_us
        self.background = background
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.audio_start_offset = audio_start_offset
        # Pre-allocate output buffer if compositing
        self.output_buffer = np.empty_like(background) if background is not None else None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=60)
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=20)
        self._process: Optional[subprocess.Popen] = None
        self._encoder_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.error: Optional[str] = None

    def _build_cmd(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24", "-r", str(self.fps), "-i", "-",
        ]
        if self.audio_start_offset > 0.0:
            cmd += ["-itsoffset", f"{self.audio_start_offset:.6f}"]
        cmd += [
            "-i", self.audio_path,
            "-c:v", "libx264", "-preset", self.preset,
            "-tune", "zerolatency",
            "-g", str(self.fps), "-keyint_min", str(self.fps),
            "-sc_threshold", "0", "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-af", "aresample=async=1:first_pts=0",
            "-flush_packets", "1",
            "-shortest",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", str(self.fragment_duration_us),
            "-f", "mp4", "pipe:1",
        ]
        return cmd

    def _encoder_worker(self):
        frame_count = 0
        try:
            while self.is_running:
                try:
                    frame = self.frame_queue.get(timeout=2.0)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                
                # Composite frame onto background if needed
                if self.background is not None:
                    frame = composite_frame(
                        frame, self.background, self.x_offset, self.y_offset, self.output_buffer
                    )
                
                if not isinstance(frame, np.ndarray) or frame.shape != (self.height, self.width, 3):
                    logger.warning(f"[{self.job_id}] Bad frame shape {getattr(frame,'shape',None)}, skipping")
                    continue
                try:
                    self._process.stdin.write(frame.tobytes())
                    frame_count += 1
                except (BrokenPipeError, Exception) as exc:
                    logger.error(f"[{self.job_id}] Encoder write error: {exc}")
                    self.error = str(exc)
                    break
        except Exception as exc:
            logger.error(f"[{self.job_id}] Encoder worker crashed: {exc}")
            self.error = str(exc)
        finally:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self.is_running = False
            logger.info(f"[{self.job_id}] Encoder done ({frame_count} frames)")

    def _reader_worker(self):
        chunk_count = 0
        try:
            while True:
                chunk = self._process.stdout.read(65536)
                if not chunk:
                    if self._process.poll() is not None:
                        break
                    continue
                self.chunk_queue.put(chunk)
                chunk_count += 1
        except Exception as exc:
            logger.error(f"[{self.job_id}] Reader worker crashed: {exc}")
            self.error = str(exc)
        finally:
            self.chunk_queue.put(None)
            logger.info(f"[{self.job_id}] Reader done ({chunk_count} chunks)")

    def start(self):
        if self.is_running:
            return
        cmd = self._build_cmd()
        logger.info(f"[{self.job_id}] Starting MP4 encoder")
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        self.is_running = True
        self._encoder_thread = threading.Thread(
            target=self._encoder_worker, daemon=True, name=f"mp4enc-{self.job_id}"
        )
        self._reader_thread = threading.Thread(
            target=self._reader_worker, daemon=True, name=f"mp4rdr-{self.job_id}"
        )
        self._encoder_thread.start()
        self._reader_thread.start()

    def add_frame(self, frame: np.ndarray):
        if not self.is_running:
            raise RuntimeError("Encoder not running")
        if self.error:
            raise RuntimeError(f"Encoder error: {self.error}")
        try:
            self.frame_queue.put(frame, timeout=2.0)
        except queue.Full:
            logger.warning(f"[{self.job_id}] Frame queue full, dropping frame")

    def get_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None  # ← CHANGED: was return b""

    def finish(self):
        self.frame_queue.put(None)
        if self._encoder_thread:
            self._encoder_thread.join(timeout=10)
        if self._reader_thread:
            self._reader_thread.join(timeout=10)
        if self._process:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def stop(self):
        self.is_running = False
        if self._process:
            self._process.terminate()