import os
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
import torch
import numpy as np
import io
import base64
import json
import asyncio
import librosa
from collections import deque
from PIL import Image
import aiohttp
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, model_validator
from typing import Literal
from loguru import logger
import uvicorn

from config import (
    MAX_CONCURRENT_STREAMS,
    FLASHHEAD_CKPT_DIR,
    WAV2VEC_DIR,
    MODEL_TYPE as DEFAULT_MODEL_TYPE,
    WARMUP_MODEL_TYPES,
    WORLD_SIZE,
    TEMP_UPLOAD_DIR,
    TEMP_OUTPUT_DIR,
    TTS_AUDIO_DIR,
    FRAGMENT_DURATION_US,
    SOULX_AVATAR_IMAGE_ROOT,
    SOULX_AVATAR_CROP_CACHE_DIR,
    SOULX_DISABLE_BACKGROUND_CACHE,
)
from avatar_image_cropper import AvatarImageCropper
from flash_head.inference import (
    get_pipeline,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
    SessionContext,
    prepare_session_base_data,
    run_pipeline_for_session,
    capture_pipeline_clean_state,
)
from session_manager import ConcurrentSessionManager, CapacityError
from video_generator import generate_mp4
from observability import generation_log_fields, log_event, traced_span
from metrics import distribution, gauge, generation_metric_tags, increment

# Global pipeline cache to keep model variants loaded
_pipelines = {}
SUPPORTED_MODEL_TYPES = {"lite", "pro"}

# Singleton session manager — created during lifespan startup so the
# asyncio.Semaphore is bound to the correct event loop.
_session_manager: ConcurrentSessionManager = None

# Serialises prepare_session_base_data calls so concurrent session setup
# does not race on the shared pipeline object (each call mutates pipeline
# attributes before snapshotting them into the SessionContext).
_setup_lock: asyncio.Lock = None
_avatar_image_cropper = AvatarImageCropper(cache_dir=SOULX_AVATAR_CROP_CACHE_DIR, target_size=512)
_avatar_image_root = Path(SOULX_AVATAR_IMAGE_ROOT)


def _resolve_avatar_image_path(avatar_id: str, use_backend_crop: bool = True) -> tuple[str, bool, bool]:
    avatar_dir = _avatar_image_root / avatar_id
    if not avatar_dir.is_dir():
        raise FileNotFoundError(f"Avatar folder not found for avatar_id='{avatar_id}': {avatar_dir}")

    candidates = []
    for name in ("avatar.png", "avatar.jpg", "avatar.jpeg"):
        candidate = avatar_dir / name
        if candidate.is_file():
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No avatar image found for avatar_id='{avatar_id}' in folder '{avatar_dir}'. Expected one of: avatar.png, avatar.jpg, avatar.jpeg"
        )

    chosen = candidates[0]
    if len(candidates) > 1:
        logger.warning(
            f"[avatar_resolve] Multiple avatar images found for avatar_id='{avatar_id}' in '{avatar_dir}'; "
            f"using preferred candidate '{chosen.name}' and ignoring {[c.name for c in candidates[1:]]}"
        )

    logger.debug(
        f"[avatar_resolve] avatar_id='{avatar_id}' avatar_dir='{avatar_dir}' chosen='{chosen}' "
        f"use_backend_crop={use_backend_crop} available={[c.name for c in candidates]}"
    )

    if not use_backend_crop:
        return str(chosen), False, False
    return _avatar_image_cropper.ensure_target_image(str(chosen), avatar_id)


def resolve_static_index() -> Path | None:
    override = os.environ.get("STATIC_INDEX_PATH")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate

    deployment_root = Path(os.environ.get("SOULX_SERVER_DIR", os.environ.get("RUNPOD_EDITS_DIR", "/opt/deployment")))
    candidates = [
        deployment_root / "static" / "index.html",
        Path("/opt/deployment/static/index.html"),
        Path("/workspace/soulx_head_static/index.html"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def normalize_model_type(model_type: str) -> str:
    normalized_model_type = (model_type or "lite").strip().lower()
    if normalized_model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: {sorted(SUPPORTED_MODEL_TYPES)}")
    return normalized_model_type

def init_pipeline(ckpt_dir: str, wav2vec_dir: str, model_type: str = "lite"):
    global _pipelines
    model_type = normalize_model_type(model_type)
    if model_type not in _pipelines:
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Invalid FLASHHEAD_CKPT_DIR: {ckpt_dir}")
        if not os.path.isdir(wav2vec_dir):
            raise FileNotFoundError(f"Invalid WAV2VEC_DIR: {wav2vec_dir}")
        log_event("pipeline_init_started", model_type=model_type, world_size=WORLD_SIZE)
        with traced_span("pipeline.init", model_type=model_type, world_size=WORLD_SIZE):
            _pipelines[model_type] = get_pipeline(
                world_size=WORLD_SIZE,
                ckpt_dir=ckpt_dir,
                wav2vec_dir=wav2vec_dir,
                model_type=model_type
            )
        # Capture clean state immediately — before any prepare_params() is ever
        # called — so prepare_session_base_data() can reset to this baseline.
        capture_pipeline_clean_state(_pipelines[model_type])
        log_event("pipeline_init_completed", model_type=model_type, world_size=WORLD_SIZE)
    return _pipelines[model_type]

async def download_audio_from_url(audio_url: str, request_suffix: str) -> str:
    """
    Download audio file from URL and save to temp directory.
    
    Args:
        audio_url: HTTP/HTTPS URL to audio file or a bare filename resolved from the local TTS audio directory
        request_suffix: Unique suffix for temp file naming
        
    Returns:
        Path to downloaded audio file
        
    Raises:
        HTTPException: If URL is invalid or download fails
    """
    normalized_audio_url = audio_url.strip()
    if not normalized_audio_url:
        raise HTTPException(status_code=400, detail="audio_url cannot be empty")

    download_start = time.time()
    log_event("audio_download_started", request_id=request_suffix, audio_url=normalized_audio_url)

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    if normalized_audio_url.startswith(('http://', 'https://')):
        resolved_audio_url = normalized_audio_url
        url_path = resolved_audio_url.split('?')[0]
        ext = os.path.splitext(url_path)[1] or '.wav'
    else:
        if '/' in normalized_audio_url or '\\' in normalized_audio_url:
            raise HTTPException(status_code=400, detail="audio_url filename must not include path separators")
        source_audio_path = os.path.join(TTS_AUDIO_DIR, normalized_audio_url)
        if not os.path.isfile(source_audio_path):
            raise HTTPException(
                status_code=400,
                detail=f"Audio file not found in TTS_AUDIO_DIR: {normalized_audio_url}"
            )
        ext = os.path.splitext(normalized_audio_url)[1] or '.wav'
        logger.info(f"Resolved audio filename '{normalized_audio_url}' to local file path: {source_audio_path}")

    if ext not in ['.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac']:
        ext = '.wav'
    
    temp_audio_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_suffix}_audio{ext}")

    if not normalized_audio_url.startswith(('http://', 'https://')):
        try:
            with open(source_audio_path, 'rb') as src, open(temp_audio_path, 'wb') as dst:
                dst.write(src.read())

            file_size = os.path.getsize(temp_audio_path)
            logger.info(f"Copied local audio file: {file_size} bytes -> {temp_audio_path}")

            if file_size == 0:
                raise HTTPException(status_code=400, detail="Local audio file is empty")

            duration_ms = round((time.time() - download_start) * 1000, 2)
            log_event("audio_download_completed", request_id=request_suffix, source="local", size_bytes=file_size, duration_ms=duration_ms)
            increment("soulx.audio.download.success")
            distribution("soulx.audio.download.duration_ms", duration_ms)
            return temp_audio_path
        except HTTPException:
            log_event("audio_download_failed", request_id=request_suffix, source="local", reason="http_exception")
            increment("soulx.audio.download.failure")
            raise
        except Exception as e:
            logger.error(f"Failed to copy local audio file: {e}")
            log_event("audio_download_failed", request_id=request_suffix, source="local", error=str(e))
            increment("soulx.audio.download.failure")
            raise HTTPException(status_code=500, detail=f"Error reading local audio file: {str(e)}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        with traced_span("audio.download", request_id=request_suffix, audio_url=resolved_audio_url):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"Downloading audio from URL: {resolved_audio_url}")
                async with session.get(resolved_audio_url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download audio from URL. HTTP {response.status}"
                        )
                    
                    content_type = response.headers.get('Content-Type', '')
                    if not any(t in content_type for t in ['audio', 'octet-stream', 'mpeg', 'wav']):
                        logger.warning(f"Unexpected content type: {content_type}")
                    
                    with open(temp_audio_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    file_size = os.path.getsize(temp_audio_path)
                    logger.info(f"Downloaded audio file: {file_size} bytes -> {temp_audio_path}")
                    
                    if file_size == 0:
                        raise HTTPException(status_code=400, detail="Downloaded audio file is empty")

                    duration_ms = round((time.time() - download_start) * 1000, 2)
                    log_event("audio_download_completed", request_id=request_suffix, source="remote", size_bytes=file_size, duration_ms=duration_ms)
                    increment("soulx.audio.download.success")
                    distribution("soulx.audio.download.duration_ms", duration_ms)
                    return temp_audio_path
                
    except aiohttp.ClientError as e:
        logger.error(f"Failed to download audio from URL: {e}")
        log_event("audio_download_failed", request_id=request_suffix, source="remote", error=str(e))
        increment("soulx.audio.download.failure")
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout downloading audio from URL: {resolved_audio_url}")
        log_event("audio_download_failed", request_id=request_suffix, source="remote", error="timeout")
        increment("soulx.audio.download.failure")
        raise HTTPException(status_code=400, detail="Timeout downloading audio from URL")
    except Exception as e:
        logger.error(f"Unexpected error downloading audio: {e}")
        log_event("audio_download_failed", request_id=request_suffix, source="remote", error=str(e))
        increment("soulx.audio.download.failure")
        raise HTTPException(status_code=500, detail=f"Error downloading audio: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session_manager

    logger.info("Server starting with warmup enabled...")
    log_event("server_starting")

    # Initialise session manager and setup lock (must happen inside the async
    # context so Semaphore/Lock are bound to the correct event loop)
    global _setup_lock
    _session_manager = ConcurrentSessionManager(max_streams=MAX_CONCURRENT_STREAMS)
    _setup_lock = asyncio.Lock()
    logger.info(f"Session manager ready: max_concurrent_streams={MAX_CONCURRENT_STREAMS}")

    # Initialize pipeline(s) at startup
    ckpt_dir = FLASHHEAD_CKPT_DIR
    wav2vec_dir = WAV2VEC_DIR
    logger.info(
        f"Initializing warmup pipeline(s): ckpt_dir={ckpt_dir}, "
        f"wav2vec_dir={wav2vec_dir}, model_types={WARMUP_MODEL_TYPES}"
    )

    warmup_models = []
    for model_type in WARMUP_MODEL_TYPES:
        try:
            model_type = normalize_model_type(model_type)
        except ValueError as exc:
            logger.error(f"Skipping invalid warmup model_type '{model_type}': {exc}")
            continue
        warmup_models.append(model_type)
        init_pipeline(ckpt_dir, wav2vec_dir, model_type)

    warmup_start = time.time()
    log_event("pipeline_warmup_started", model_types=warmup_models)

    try:
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        dummy_img_path = os.path.join(TEMP_UPLOAD_DIR, "dummy_warmup.jpg")
        if not os.path.exists(dummy_img_path):
            dummy_img = Image.new('RGB', (512, 512), color='black')
            dummy_img.save(dummy_img_path)

        for model_type in warmup_models:
            per_model_start = time.time()
            log_event("pipeline_warmup_model_started", model_type=model_type)
            with traced_span("pipeline.warmup", model_type=model_type):
                pipeline = init_pipeline(ckpt_dir, wav2vec_dir, model_type)
                ctx = SessionContext(
                    session_id=f"startup_warmup_{model_type}",
                    cond_image_path_or_dir=dummy_img_path,
                    base_seed=42,
                    use_face_crop=False,
                    expression_scale=1.0,
                )
                prepare_session_base_data(pipeline, ctx)

                infer_params = get_infer_params()
                sample_rate = infer_params['sample_rate']
                tgt_fps = infer_params['tgt_fps']
                frame_num = infer_params['frame_num']
                cached_audio_duration = infer_params['cached_audio_duration']

                dummy_audio_length = sample_rate * cached_audio_duration
                dummy_audio = np.zeros(dummy_audio_length, dtype=np.float32)

                audio_end_idx = cached_audio_duration * tgt_fps
                audio_start_idx = audio_end_idx - frame_num
                dummy_embedding = get_audio_embedding(pipeline, dummy_audio, audio_start_idx, audio_end_idx)
                _ = run_pipeline(pipeline, dummy_embedding)
                _ = run_pipeline_for_session(pipeline, ctx, dummy_embedding)

                # Warm up librosa MP3 decode path to avoid 14s cold-start on first request
                librosa_warmup_start = time.time()
                try:
                    import librosa as _librosa
                    dummy_mp3_path = os.path.join(TEMP_UPLOAD_DIR, "dummy_warmup.mp3")
                    if not os.path.exists(dummy_mp3_path):
                        import subprocess as _sp
                        _sp.run([
                            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                            "-t", "0.5", "-codec:a", "libmp3lame", "-q:a", "9", dummy_mp3_path
                        ], check=True, capture_output=True)
                    _librosa.load(dummy_mp3_path, sr=16000)
                    logger.info(
                        f"Warmup librosa MP3 decode complete in "
                        f"{(time.time() - librosa_warmup_start):.2f}s"
                    )
                except Exception as _e:
                    logger.warning(f"Warmup librosa MP3 decode skipped: {_e}")

                dummy_audio_path = os.path.join(TEMP_UPLOAD_DIR, "dummy_warmup.wav")
                if not os.path.exists(dummy_audio_path):
                    import wave
                    with wave.open(dummy_audio_path, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(np.zeros(sample_rate // 2, dtype=np.int16).tobytes())

                from mp4_stream_encoder import Mp4StreamEncoder
                encoder_start = time.time()
                encoder = Mp4StreamEncoder(
                    width=512,
                    height=512,
                    fps=tgt_fps,
                    audio_path=dummy_audio_path,
                    job_id=f"startup_encoder_warmup_{model_type}",
                    fragment_duration_us=FRAGMENT_DURATION_US,
                )
                encoder.start()
                black_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                for _ in range(max(1, int(tgt_fps // 2))):
                    encoder.add_frame(black_frame)
                encoder.finish()
                while encoder.get_chunk(timeout=0.1) is not None:
                    pass
                logger.info(
                    f"Warmup fMP4 encoder for model_type={model_type} complete in "
                    f"{(time.time() - encoder_start):.2f}s"
                )

            per_model_duration = time.time() - per_model_start
            logger.info(
                f"Warmup for model_type={model_type} complete in {per_model_duration:.2f}s"
            )
            log_event(
                "pipeline_warmup_model_completed",
                model_type=model_type,
                duration_ms=round(per_model_duration * 1000, 2),
                status="success",
            )
            distribution(
                "soulx.pipeline.warmup.duration_ms",
                round(per_model_duration * 1000, 2),
                tags=generation_metric_tags("startup", model_type, status="success"),
            )

        warmup_duration = time.time() - warmup_start
        logger.info(
            f"Warmup complete in {warmup_duration:.2f}s - "
            f"models={warmup_models} compiled and ready!"
        )
        log_event(
            "pipeline_warmup_completed",
            model_types=warmup_models,
            duration_ms=round(warmup_duration * 1000, 2),
            status="success",
        )

        import observability
        observability.server_started = True
        logger.info("Server marked as READY - warmup successful")

    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        log_event("pipeline_warmup_failed", model_types=warmup_models, error=str(e), status="failure")
        # Keep server_started = False so health checks fail and ALB doesn't route traffic here

    yield

    logger.info("Server shutting down...")
    log_event("server_shutting_down")


class ErrorResponse(BaseModel):
    error: str


class CapacityErrorResponse(BaseModel):
    error: str
    message: str


class HealthResponse(BaseModel):
    success: bool
    configured: bool
    runpodStatus: str
    apiUrl: str


class AnalyticsDashboardData(BaseModel):
    totalVideos: int
    totalMinutes: int
    averageGenerationTime: int | float
    successRate: int | float
    recentActivity: list


class AnalyticsDashboardResponse(BaseModel):
    success: bool
    data: AnalyticsDashboardData


class SessionStatus(BaseModel):
    session_id: str
    age_seconds: float
    frames_generated: int
    buffered_samples: int
    buffered_ms: int
    total_received: int
    total_chunks: int


class ConcurrencyStatusResponse(BaseModel):
    active_streams: int
    capacity: int
    available: int
    at_capacity: bool
    sessions: list[SessionStatus]


EXPRESSION_SCALE_MIN = 0.65
EXPRESSION_SCALE_MAX = 1.35


class GenerateArgs(BaseModel):
    model_type: Literal["lite", "pro"] = "lite"
    base_seed: int = 42
    use_face_crop: bool = False
    silence_padding_sec: float = 0.0
    fragment_duration_ms: int | None = None
    preserve_aspect_ratio: bool = False
    expression_scale: float = 1.0
    output_scale: int = 1
    encoder_crf: int = 20
    encoder_preset: str = "veryfast"
    jpeg_quality: int = 85

    @model_validator(mode='after')
    def validate_mutually_exclusive_flags(self):
        if self.use_face_crop and self.preserve_aspect_ratio:
            raise ValueError(
                "use_face_crop and preserve_aspect_ratio cannot both be true. "
                "Use use_face_crop=true for face-focused square videos, "
                "or use_face_crop=false + preserve_aspect_ratio=true for aspect-ratio-preserved videos."
            )
        if not EXPRESSION_SCALE_MIN <= self.expression_scale <= EXPRESSION_SCALE_MAX:
            raise ValueError(f"expression_scale must be between {EXPRESSION_SCALE_MIN} and {EXPRESSION_SCALE_MAX}")
        return self


def _model_output_side() -> int:
    infer_params = get_infer_params()
    return int(infer_params.get("height", 512))


class StreamEfsRequest(BaseModel):
    avatar_id: str
    audio: str
    args: str = "{}"


def _parse_args(args_str: str) -> GenerateArgs:
    try:
        data = json.loads(args_str or "{}")
        return GenerateArgs(**{k: v for k, v in data.items() if k in GenerateArgs.model_fields})
    except (json.JSONDecodeError, TypeError):
        return GenerateArgs()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _resolve_generation_args(args: GenerateArgs) -> GenerateArgs:
    args.output_scale = _validate_output_scale(args.output_scale)
    args.encoder_preset = _validate_encoder_preset(args.encoder_preset)
    args.encoder_crf = _validate_encoder_crf(args.encoder_crf)
    args.jpeg_quality = _validate_jpeg_quality(args.jpeg_quality)
    return args


app = FastAPI(
    title="SoulX-FlashHead Streaming API",
    description="API for generating SoulX avatar video output via MJPEG streaming, fragmented MP4 streaming, full MP4 generation, health monitoring, and concurrency status inspection.",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.middleware("http")
async def observability_http_middleware(request: Request, call_next):
    status_paths = {"/api/status", "/status"}
    if (
        request.url.path in status_paths
        and _session_manager is not None
        and _session_manager.active_count() > 0
    ):
        return await call_next(request)

    request_id = request.headers.get("x-request-id", f"http_{uuid.uuid4().hex[:12]}")
    start_time = time.time()
    log_event("http_request_started", request_id=request_id, method=request.method, path=request.url.path)

    try:
        with traced_span("http.request", request_id=request_id, method=request.method, path=request.url.path):
            response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log_event("http_request_completed", request_id=request_id, method=request.method, path=request.url.path, status_code=response.status_code, duration_ms=duration_ms)
        return response
    except Exception as exc:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_event("http_request_failed", request_id=request_id, method=request.method, path=request.url.path, duration_ms=duration_ms, error=str(exc))
        raise

_static_index = resolve_static_index()
_static_dir = _static_index.parent if _static_index else None

if _static_dir and _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    _assets_dir = _static_dir / "assets"
    _videos_dir = _static_dir / "videos"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")
    if _videos_dir.exists():
        app.mount("/videos", StaticFiles(directory=str(_videos_dir)), name="videos")

def tensor_to_jpeg_bytes(tensor_frame):
    """Convert a single frame tensor (H, W, C) to JPEG bytes."""
    # tensor_frame is expected to be shape (H, W, C) with values 0-255
    frame_np = tensor_frame.numpy().astype(np.uint8)
    img = Image.fromarray(frame_np)
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _validate_output_scale(output_scale: int) -> int:
    if not isinstance(output_scale, int) or output_scale < 1 or output_scale > 4:
        raise ValueError("output_scale must be an integer from 1 to 4")
    return output_scale


def _validate_encoder_preset(preset: str) -> str:
    allowed = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium"}
    normalized = (preset or "veryfast").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"encoder_preset must be one of: {sorted(allowed)}")
    return normalized


def _validate_encoder_crf(crf: int) -> int:
    if not isinstance(crf, int) or crf < 0 or crf > 51:
        raise ValueError("encoder_crf must be an integer from 0 to 51")
    return crf


def _validate_jpeg_quality(quality: int) -> int:
    if not isinstance(quality, int) or quality < 1 or quality > 95:
        raise ValueError("jpeg_quality must be an integer from 1 to 95")
    return quality


def _encode_jpeg(frame_np: np.ndarray, quality: int) -> bytes:
    img = Image.fromarray(frame_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def _validate_audio_inputs(audio: UploadFile | None, audio_url: str | None):
    if (audio is None and audio_url is None) or (audio is not None and audio_url is not None):
        return JSONResponse(
            status_code=400,
            content={"error": "Exactly one of 'audio' (file) or 'audio_url' (string) must be provided"}
        )
    return None

def _validate_driving_inputs(driving: UploadFile | None, avatar_id: str | None):
    normalized_avatar_id = (avatar_id or "").strip()
    if (driving is None and not normalized_avatar_id) or (driving is not None and normalized_avatar_id):
        return JSONResponse(
            status_code=400,
            content={"error": "Exactly one of 'driving' (file) or 'avatar_id' (string) must be provided"}
        )
    return None

async def _resolve_driving_image_path(
    request_suffix: str,
    driving: UploadFile | None,
    avatar_id: str | None,
    use_face_crop: bool,
    preserve_aspect_ratio: bool = False,
):
    normalized_avatar_id = (avatar_id or "").strip()
    if normalized_avatar_id:
        try:
            # When preserve_aspect_ratio is true, skip center crop to preserve original dimensions
            use_backend_crop = not use_face_crop and not preserve_aspect_ratio
            resolved_path, was_cropped, was_cached = _resolve_avatar_image_path(
                normalized_avatar_id,
                use_backend_crop=use_backend_crop,
            )
            if use_face_crop:
                logger.info(
                    f"[{request_suffix}] Resolved avatar_id={normalized_avatar_id} with use_face_crop enabled; skipping backend center crop and using source image {resolved_path}"
                )
            elif preserve_aspect_ratio:
                logger.info(
                    f"[{request_suffix}] Resolved avatar_id={normalized_avatar_id} with preserve_aspect_ratio enabled; skipping backend center crop and using original image {resolved_path}"
                )
            elif was_cropped and was_cached:
                logger.info(
                    f"[{request_suffix}] Resolved avatar_id={normalized_avatar_id} using cached center crop -> {resolved_path}"
                )
            elif was_cropped:
                logger.info(
                    f"[{request_suffix}] Resolved avatar_id={normalized_avatar_id}; source image was not 512x512 so a center crop was created and cached at {resolved_path}"
                )
            else:
                logger.info(f"[{request_suffix}] Resolved avatar_id={normalized_avatar_id} -> {resolved_path}")
            return resolved_path
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

    if driving is None:
        return JSONResponse(status_code=400, content={"error": "Driving image is required"})

    temp_driving_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_suffix}_{driving.filename}")
    driving_data = await driving.read()
    try:
        img = Image.open(io.BytesIO(driving_data))
        width, height = img.size
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Invalid image format: {str(exc)}. Please upload a valid image file."})

    with open(temp_driving_path, "wb") as f:
        f.write(driving_data)

    if use_face_crop:
        logger.info(
            f"[{request_suffix}] Uploaded driving image is {width}x{height}; use_face_crop enabled so backend center crop is skipped and inference-time face crop will be used"
        )
        return temp_driving_path

    resolved_path, was_cropped, was_cached = _avatar_image_cropper.ensure_target_image(
        temp_driving_path,
        f"upload:{request_suffix}"
    )
    if was_cropped and was_cached:
        logger.info(
            f"[{request_suffix}] Uploaded driving image resolved using cached center crop -> {resolved_path}"
        )
    elif was_cropped:
        logger.info(
            f"[{request_suffix}] Uploaded driving image was {width}x{height}; created and cached centered 512x512 crop at {resolved_path}"
        )
    else:
        logger.info(f"[{request_suffix}] Uploaded driving image validation passed: {width}x{height}px")

    return resolved_path

async def _stream_logger_wrapper(gen, endpoint, request_id, client_ip, media_info, args_dict):
    import inspect
    from observability import log_request_summary
    start_time = time.time()
    try:
        if inspect.isasyncgen(gen):
            async for chunk in gen:
                yield chunk
        else:
            for chunk in gen:
                yield chunk
        duration_ms = (time.time() - start_time) * 1000
        log_request_summary(
            endpoint=endpoint,
            request_id=request_id,
            client_ip=client_ip,
            media_info=media_info,
            args=args_dict,
            status="COMPLETED",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        log_request_summary(
            endpoint=endpoint,
            request_id=request_id,
            client_ip=client_ip,
            media_info=media_info,
            args=args_dict,
            status="FAILED",
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise

@app.post(
    "/generate-dfs",
    summary="Generate MJPEG stream (DFS)",
    description="Streams generated video frames as MJPEG (Direct Frame Streaming). Provide exactly one of `driving` or `avatar_id`, and exactly one of `audio` or `audio_url`.",
    responses={
        200: {
            "description": "MJPEG multipart frame stream.",
            "content": {
                "multipart/x-mixed-replace; boundary=frame": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters or media input."},
        503: {"model": CapacityErrorResponse, "description": "Server is at concurrent stream capacity."},
    },
)
async def generate_stream(
    request: Request,
    driving: UploadFile = File(None, description="Driving image upload. Mutually exclusive with `avatar_id`."),
    avatar_id: str = Form(None, description="Preconfigured avatar identifier. Mutually exclusive with `driving`."),
    audio: UploadFile = File(None, description="Audio file upload. Mutually exclusive with `audio_url`."),
    audio_url: str = Form(None, description="HTTP/HTTPS URL to an audio file. Mutually exclusive with `audio`."),
    model_type: Literal["lite", "pro"] = Form("lite", description="Model variant to use."),
    base_seed: int = Form(42, description="Seed used for deterministic generation behavior."),
    use_face_crop: bool = Form(False, description="If true, apply inference-time face crop handling instead of backend center crop."),
    silence_padding_sec: float = Form(0.0, description="Number of seconds of silence to prepend before generation begins."),
):
    request_start = time.time()
    audio_validation_error = _validate_audio_inputs(audio, audio_url)
    if audio_validation_error is not None:
        return audio_validation_error
    driving_validation_error = _validate_driving_inputs(driving, avatar_id)
    if driving_validation_error is not None:
        return driving_validation_error

    if _session_manager is None or _session_manager.is_at_capacity():
        return JSONResponse(
            status_code=503,
            content={
                "error": "CAPACITY_FULL",
                "message": (
                    f"Server is at capacity ({MAX_CONCURRENT_STREAMS} concurrent streams). "
                    "Please try again later."
                ),
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    request_suffix = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"
    
    if audio_url:
        logger.info(f"Using audio from URL: {audio_url}")
        temp_audio_path = await download_audio_from_url(audio_url, request_suffix)
    else:
        temp_audio_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_suffix}_{audio.filename}")
    resolved_driving_path = await _resolve_driving_image_path(request_suffix, driving, avatar_id, use_face_crop, False)
    if isinstance(resolved_driving_path, JSONResponse):
        return resolved_driving_path
    
    if audio:
        with open(temp_audio_path, "wb") as f:
            f.write(await audio.read())

    try:
        model_type = normalize_model_type(model_type)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    tags = generation_metric_tags("dfs_stream", model_type, endpoint="generate-dfs")
    log_event("generation_request_received", **generation_log_fields(request_suffix, "dfs_stream", model_type, avatar_id=avatar_id or None, has_audio_url=bool(audio_url), use_face_crop=use_face_crop, base_seed=base_seed))
    increment("soulx.generation.requests", tags=tags)
    init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)
    log_event("generation_stream_response_started", **generation_log_fields(request_suffix, "dfs_stream", model_type, duration_ms=round((time.time() - request_start) * 1000, 2)))

    media_info = {
        "avatar_id": avatar_id or None,
        "driving": driving.filename if driving else None,
        "audio": audio.filename if audio else None,
        "audio_url": audio_url or None,
    }
    args_dict = {
        "model_type": model_type,
        "base_seed": base_seed,
        "use_face_crop": use_face_crop,
        "silence_padding_sec": silence_padding_sec,
    }
    from observability import log_request_summary
    log_request_summary(
        endpoint="/generate-dfs",
        request_id=request_suffix,
        client_ip=request.client.host if request.client else "unknown",
        media_info=media_info,
        args=args_dict,
        status="STARTED",
    )

    return StreamingResponse(
        _stream_logger_wrapper(
            generate_frames_stream(
                resolved_driving_path,
                temp_audio_path,
                base_seed,
                use_face_crop,
                model_type,
                silence_padding_sec,
                request_id=request_suffix,
            ),
            endpoint="/generate-dfs",
            request_id=request_suffix,
            client_ip=request.client.host if request.client else "unknown",
            media_info=media_info,
            args_dict=args_dict,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post(
    "/generate-mpeg-stream",
    summary="Generate fragmented MP4 stream",
    description="Streams generated video as fragmented MP4 with audio muxed server-side. Provide exactly one of `driving` or `avatar_id`, and exactly one of `audio` or `audio_url`.",
    responses={
        200: {
            "description": "Fragmented MP4 stream.",
            "content": {
                "video/mp4": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters or media input."},
        503: {"model": CapacityErrorResponse, "description": "Server is at concurrent stream capacity."},
    },
)
async def generate_mpeg_stream(
    request: Request,
    driving: UploadFile = File(None, description="Driving image upload. Mutually exclusive with `avatar_id`."),
    avatar_id: str = Form(None, description="Preconfigured avatar identifier. Mutually exclusive with `driving`."),
    audio: UploadFile = File(None, description="Audio file upload. Mutually exclusive with `audio_url`."),
    audio_url: str = Form(None, description="HTTP/HTTPS URL to an audio file. Mutually exclusive with `audio`."),
    model_type: Literal["lite", "pro"] = Form("lite", description="Model variant to use."),
    base_seed: int = Form(42, description="Seed used for deterministic generation behavior."),
    use_face_crop: bool = Form(False, description="If true, apply inference-time face crop handling instead of backend center crop."),
    silence_padding_sec: float = Form(0.0, description="Number of seconds of silence to prepend before generation begins."),
    fragment_duration_ms: int = Form(None, description="Optional MP4 fragment duration in milliseconds. If omitted, server default is used."),
    preserve_aspect_ratio: bool = Form(False, description="If true, preserve input image aspect ratio by compositing 512x512 frames onto original background."),
    expression_scale: float = Form(1.0, ge=EXPRESSION_SCALE_MIN, le=EXPRESSION_SCALE_MAX, description="Intensity of lip sync and head movement."),
):
    """
    Stream a fragmented MP4 (fMP4) with server-side audio muxing.
    Bypasses DFS — FFmpeg encodes frames + audio in real time and the
    browser can start playback as soon as the first fragment arrives.
    """
    from mp4_stream_handlers import generate_mp4_stream

    request_start = time.time()
    audio_validation_error = _validate_audio_inputs(audio, audio_url)
    if audio_validation_error is not None:
        return audio_validation_error
    driving_validation_error = _validate_driving_inputs(driving, avatar_id)
    if driving_validation_error is not None:
        return driving_validation_error

    # Validate mutually exclusive flags
    if use_face_crop and preserve_aspect_ratio:
        return JSONResponse(
            status_code=400,
            content={
                "error": "use_face_crop and preserve_aspect_ratio cannot both be true. "
                        "Use use_face_crop=true for face-focused square videos, "
                        "or use_face_crop=false + preserve_aspect_ratio=true for aspect-ratio-preserved videos."
            }
        )

    if _session_manager is None or _session_manager.is_at_capacity():
        return JSONResponse(
            status_code=503,
            content={
                "error": "CAPACITY_FULL",
                "message": (
                    f"Server is at capacity ({MAX_CONCURRENT_STREAMS} concurrent streams). "
                    "Please try again later."
                ),
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    request_suffix = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"

    if audio_url:
        logger.info(f"Using audio from URL: {audio_url}")
        temp_audio_path = await download_audio_from_url(audio_url, request_suffix)
    else:
        temp_audio_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_suffix}_{audio.filename}")
    resolved_driving_path = await _resolve_driving_image_path(request_suffix, driving, avatar_id, use_face_crop, preserve_aspect_ratio)
    if isinstance(resolved_driving_path, JSONResponse):
        error_body = json.loads(resolved_driving_path.body.decode("utf-8"))
        message = error_body.get("error", "Invalid driving input")
        if avatar_id:
            return JSONResponse(status_code=400, content={"error": message})
        if "dimensions" in message:
            model_side = _model_output_side()
            return JSONResponse(status_code=400, content={"error": message.replace(f"Driving images must be exactly {model_side}x{model_side}px.", f"Must be {model_side}x{model_side}px.")})
        if message.startswith("Invalid image format:"):
            return JSONResponse(status_code=400, content={"error": message.replace("Invalid image format: ", "Invalid image: ").replace(". Please upload a valid image file.", "")})
        return JSONResponse(status_code=400, content={"error": message})

    if audio:
        with open(temp_audio_path, "wb") as f:
            f.write(await audio.read())

    try:
        model_type = normalize_model_type(model_type)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    tags = generation_metric_tags("mpeg_stream", model_type, endpoint="generate-mpeg-stream")
    log_event("generation_request_received", **generation_log_fields(request_suffix, "mpeg_stream", model_type, avatar_id=avatar_id or None, has_audio_url=bool(audio_url), use_face_crop=use_face_crop, base_seed=base_seed))
    increment("soulx.generation.requests", tags=tags)
    init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)

    frag_us = (fragment_duration_ms * 1000) if fragment_duration_ms is not None else FRAGMENT_DURATION_US
    log_event("generation_stream_response_started", **generation_log_fields(request_suffix, "mpeg_stream", model_type, fragment_duration_us=frag_us, duration_ms=round((time.time() - request_start) * 1000, 2)))

    media_info = {
        "avatar_id": avatar_id or None,
        "driving": driving.filename if driving else None,
        "audio": audio.filename if audio else None,
        "audio_url": audio_url or None,
    }
    args_dict = {
        "model_type": model_type,
        "base_seed": base_seed,
        "use_face_crop": use_face_crop,
        "silence_padding_sec": silence_padding_sec,
        "fragment_duration_ms": fragment_duration_ms,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "expression_scale": expression_scale,
    }
    from observability import log_request_summary
    log_request_summary(
        endpoint="/generate-mpeg-stream",
        request_id=request_suffix,
        client_ip=request.client.host if request.client else "unknown",
        media_info=media_info,
        args=args_dict,
        status="STARTED",
    )

    return StreamingResponse(
        _stream_logger_wrapper(
            generate_mp4_stream(
                resolved_driving_path,
                temp_audio_path,
                base_seed,
                use_face_crop,
                model_type,
                silence_padding_sec,
                output_scale=1,
                encoder_crf=20,
                encoder_preset="veryfast",
                fragment_duration_us=frag_us,
                request_id=request_suffix,
                preserve_aspect_ratio=preserve_aspect_ratio,
                expression_scale=expression_scale,
            ),
            endpoint="/generate-mpeg-stream",
            request_id=request_suffix,
            client_ip=request.client.host if request.client else "unknown",
            media_info=media_info,
            args_dict=args_dict,
        ),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post(
    "/stream-efs",
    summary="Generate fMP4 stream from EFS/URL audio (JSON)",
    description="Streams avatar video as fragmented MP4. Accepts a JSON body with `avatar_id`, `audio` (bare filename resolved from EFS/TTS_AUDIO_DIR or HTTP/HTTPS URL), and an optional `args` JSON string that can include `model_type`, `base_seed`, `use_face_crop`, `silence_padding_sec`, `fragment_duration_ms`, and `expression_scale`.",
    responses={
        200: {
            "description": "Fragmented MP4 stream.",
            "content": {
                "video/mp4": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "headers": {
                "X-Target-FPS": {
                    "description": "Target encoded video frame rate for this stream.",
                    "schema": {"type": "string"},
                },
                "X-Fragment-Duration-MS": {
                    "description": "Resolved fMP4 fragment duration in milliseconds for this stream.",
                    "schema": {"type": "string"},
                },
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters or media input."},
        503: {"model": CapacityErrorResponse, "description": "Server is at concurrent stream capacity."},
    },
)
async def stream_efs(request: StreamEfsRequest, http_request: Request):
    from mp4_stream_handlers import generate_mp4_stream
    from flash_head.inference import get_infer_params

    request_start = time.time()
    if _session_manager is None or _session_manager.is_at_capacity():
        return JSONResponse(
            status_code=503,
            content={
                "error": "CAPACITY_FULL",
                "message": (
                    f"Server is at capacity ({MAX_CONCURRENT_STREAMS} concurrent streams). "
                    "Please try again later."
                ),
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    args = _resolve_generation_args(_parse_args(request.args))

    try:
        model_type = normalize_model_type(args.model_type)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    request_suffix = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"

    try:
        audio_dl_start = time.time()
        temp_audio_path = await download_audio_from_url(request.audio, request_suffix)
        logger.info(
            f"[{request_suffix}] stream_efs audio download took "
            f"{(time.time() - audio_dl_start) * 1000:.1f}ms"
        )
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    avatar_start = time.time()
    resolved_driving_path = await _resolve_driving_image_path(
        request_suffix, None, request.avatar_id, args.use_face_crop, args.preserve_aspect_ratio
    )
    if isinstance(resolved_driving_path, JSONResponse):
        return resolved_driving_path
    logger.info(
        f"[{request_suffix}] stream_efs avatar resolve took "
        f"{(time.time() - avatar_start) * 1000:.1f}ms"
    )

    if args.preserve_aspect_ratio:
        logger.debug(
            f"[{request_suffix}] stream_efs preserve_aspect_ratio={args.preserve_aspect_ratio} "
            f"background_cache_disabled={SOULX_DISABLE_BACKGROUND_CACHE} "
            f"resolved_driving_path={resolved_driving_path}"
        )

    pipeline_init_start = time.time()
    init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)
    logger.info(
        f"[{request_suffix}] stream_efs pipeline init took "
        f"{(time.time() - pipeline_init_start) * 1000:.1f}ms"
    )
    infer_params = get_infer_params()
    target_fps = infer_params["tgt_fps"]
    fragment_duration_ms = args.fragment_duration_ms if args.fragment_duration_ms is not None else (FRAGMENT_DURATION_US // 1000)
    fragment_duration_us = fragment_duration_ms * 1000
    tags = generation_metric_tags("stream_efs", model_type, endpoint="stream-efs")
    log_event("generation_request_received", **generation_log_fields(request_suffix, "stream_efs", model_type, avatar_id=request.avatar_id, has_audio_url=True, use_face_crop=args.use_face_crop, base_seed=args.base_seed))
    log_event("generation_stream_response_started", **generation_log_fields(request_suffix, "stream_efs", model_type, fragment_duration_us=fragment_duration_us, duration_ms=round((time.time() - request_start) * 1000, 2)))
    increment("soulx.generation.requests", tags=tags)

    media_info = {
        "avatar_id": request.avatar_id or None,
        "audio": request.audio or None,
    }
    args_dict = {
        "model_type": model_type,
        "base_seed": args.base_seed,
        "use_face_crop": args.use_face_crop,
        "silence_padding_sec": args.silence_padding_sec,
        "fragment_duration_ms": fragment_duration_ms,
        "preserve_aspect_ratio": args.preserve_aspect_ratio,
        "expression_scale": args.expression_scale,
    }
    from observability import log_request_summary
    log_request_summary(
        endpoint="/stream-efs",
        request_id=request_suffix,
        client_ip=http_request.client.host if http_request.client else "unknown",
        media_info=media_info,
        args=args_dict,
        status="STARTED",
    )

    return StreamingResponse(
        _stream_logger_wrapper(
            generate_mp4_stream(
                resolved_driving_path,
                temp_audio_path,
                args.base_seed,
                args.use_face_crop,
                model_type,
                args.silence_padding_sec,
                output_scale=args.output_scale,
                encoder_crf=args.encoder_crf,
                encoder_preset=args.encoder_preset,
                fragment_duration_us=fragment_duration_us,
                request_id=request_suffix,
                preserve_aspect_ratio=args.preserve_aspect_ratio,
                expression_scale=args.expression_scale,
            ),
            endpoint="/stream-efs",
            request_id=request_suffix,
            client_ip=http_request.client.host if http_request.client else "unknown",
            media_info=media_info,
            args_dict=args_dict,
        ),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Target-FPS": str(target_fps),
            "X-Fragment-Duration-MS": str(fragment_duration_ms),
        },
    )


@app.post(
    "/generate-file",
    summary="Generate fMP4 stream from uploaded audio file",
    description="Streams avatar video as fragmented MP4. Accepts multipart form with `avatar_id` or `driving` image, an `audio` file upload, and an optional `args` JSON string that can include `model_type`, `base_seed`, `use_face_crop`, `silence_padding_sec`, and `expression_scale`.",
    responses={
        200: {
            "description": "Fragmented MP4 stream.",
            "content": {
                "video/mp4": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters or media input."},
        503: {"model": CapacityErrorResponse, "description": "Server is at concurrent stream capacity."},
    },
)
async def generate_file(
    request: Request,
    driving: UploadFile = File(None, description="Driving image upload. Mutually exclusive with `avatar_id`."),
    avatar_id: str = Form(None, description="Preconfigured avatar identifier. Mutually exclusive with `driving`."),
    audio: UploadFile = File(..., description="Audio file upload."),
    args: str = Form("{}", description="JSON string of generation args: model_type, base_seed, use_face_crop, silence_padding_sec."),
):
    from mp4_stream_handlers import generate_mp4_stream

    request_start = time.time()
    driving_validation_error = _validate_driving_inputs(driving, avatar_id)
    if driving_validation_error is not None:
        return driving_validation_error

    if _session_manager is None or _session_manager.is_at_capacity():
        return JSONResponse(
            status_code=503,
            content={
                "error": "CAPACITY_FULL",
                "message": (
                    f"Server is at capacity ({MAX_CONCURRENT_STREAMS} concurrent streams). "
                    "Please try again later."
                ),
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    parsed_args = _parse_args(args)

    try:
        model_type = normalize_model_type(parsed_args.model_type)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    request_suffix = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"

    temp_audio_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_suffix}_{audio.filename}")
    with open(temp_audio_path, "wb") as f:
        f.write(await audio.read())

    resolved_driving_path = await _resolve_driving_image_path(
        request_suffix, driving, avatar_id, parsed_args.use_face_crop, parsed_args.preserve_aspect_ratio
    )
    if isinstance(resolved_driving_path, JSONResponse):
        return resolved_driving_path

    init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)
    tags = generation_metric_tags("generate_file", model_type, endpoint="generate-file")
    log_event("generation_request_received", **generation_log_fields(request_suffix, "generate_file", model_type, avatar_id=avatar_id or None, has_audio_url=False, use_face_crop=parsed_args.use_face_crop, base_seed=parsed_args.base_seed))
    log_event("generation_stream_response_started", **generation_log_fields(request_suffix, "generate_file", model_type, duration_ms=round((time.time() - request_start) * 1000, 2)))
    increment("soulx.generation.requests", tags=tags)

    media_info = {
        "avatar_id": avatar_id or None,
        "driving": driving.filename if driving else None,
        "audio": audio.filename if audio else None,
    }
    args_dict = {
        "model_type": model_type,
        "base_seed": parsed_args.base_seed,
        "use_face_crop": parsed_args.use_face_crop,
        "silence_padding_sec": parsed_args.silence_padding_sec,
        "fragment_duration_ms": (FRAGMENT_DURATION_US // 1000),
        "preserve_aspect_ratio": parsed_args.preserve_aspect_ratio,
        "expression_scale": parsed_args.expression_scale,
    }
    from observability import log_request_summary
    log_request_summary(
        endpoint="/generate-file",
        request_id=request_suffix,
        client_ip=request.client.host if request.client else "unknown",
        media_info=media_info,
        args=args_dict,
        status="STARTED",
    )

    return StreamingResponse(
        _stream_logger_wrapper(
            generate_mp4_stream(
                resolved_driving_path,
                temp_audio_path,
                parsed_args.base_seed,
                parsed_args.use_face_crop,
                model_type,
                parsed_args.silence_padding_sec,
                fragment_duration_us=FRAGMENT_DURATION_US,
                request_id=request_suffix,
                preserve_aspect_ratio=parsed_args.preserve_aspect_ratio,
                expression_scale=parsed_args.expression_scale,
            ),
            endpoint="/generate-file",
            request_id=request_suffix,
            client_ip=request.client.host if request.client else "unknown",
            media_info=media_info,
            args_dict=args_dict,
        ),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post(
    "/generate_video",
    summary="Generate complete MP4 file",
    description="Generates the full video and returns a downloadable MP4 file. Provide exactly one of `driving` or `avatar_id`, and exactly one of `audio` or `audio_url`.",
    responses={
        200: {
            "description": "Generated MP4 file.",
            "content": {
                "video/mp4": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters or media input."},
    },
)
async def generate_video(
    request: Request,
    driving: UploadFile = File(None, description="Driving image upload. Mutually exclusive with `avatar_id`."),
    avatar_id: str = Form(None, description="Preconfigured avatar identifier. Mutually exclusive with `driving`."),
    audio: UploadFile = File(None, description="Audio file upload. Mutually exclusive with `audio_url`."),
    audio_url: str = Form(None, description="HTTP/HTTPS URL to an audio file. Mutually exclusive with `audio`."),
    model_type: Literal["lite", "pro"] = Form("lite", description="Model variant to use."),
    base_seed: int = Form(42, description="Seed used for deterministic generation behavior."),
    use_face_crop: bool = Form(False, description="If true, apply inference-time face crop handling instead of backend center crop."),
    silence_padding_sec: float = Form(0.0, description="Number of seconds of silence to prepend before generation begins."),
    preserve_aspect_ratio: bool = Form(False, description="If true, preserve input image aspect ratio by compositing 512x512 frames onto original background."),
    expression_scale: float = Form(1.0, ge=EXPRESSION_SCALE_MIN, le=EXPRESSION_SCALE_MAX, description="Intensity of lip sync and head movement."),
):
    request_start = time.time()
    audio_validation_error = _validate_audio_inputs(audio, audio_url)
    if audio_validation_error is not None:
        return audio_validation_error
    driving_validation_error = _validate_driving_inputs(driving, avatar_id)
    if driving_validation_error is not None:
        return driving_validation_error

    # Validate mutually exclusive flags
    if use_face_crop and preserve_aspect_ratio:
        return JSONResponse(
            status_code=400,
            content={
                "error": "use_face_crop and preserve_aspect_ratio cannot both be true. "
                        "Use use_face_crop=true for face-focused square videos, "
                        "or use_face_crop=false + preserve_aspect_ratio=true for aspect-ratio-preserved videos."
            }
        )

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    ts = int(time.time())
    request_suffix = f"vid_{ts}"

    if audio_url:
        logger.info(f"Using audio from URL: {audio_url}")
        temp_audio_path = await download_audio_from_url(audio_url, request_suffix)
    else:
        temp_audio_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_suffix}_{audio.filename}")

    resolved_driving_path = await _resolve_driving_image_path(request_suffix, driving, avatar_id, use_face_crop, preserve_aspect_ratio)
    if isinstance(resolved_driving_path, JSONResponse):
        error_body = json.loads(resolved_driving_path.body.decode("utf-8"))
        return {"error": error_body.get("error", "Invalid driving input")}

    if preserve_aspect_ratio:
        logger.debug(
            f"[{request_suffix}] generate_video preserve_aspect_ratio={preserve_aspect_ratio} "
            f"background_cache_disabled={SOULX_DISABLE_BACKGROUND_CACHE} "
            f"resolved_driving_path={resolved_driving_path}"
        )

    if audio:
        with open(temp_audio_path, "wb") as f:
            f.write(await audio.read())

    try:
        model_type = normalize_model_type(model_type)
    except ValueError as exc:
        return {"error": str(exc)}

    tags = generation_metric_tags("mp4_file", model_type, endpoint="generate_video")
    log_event("generation_started", **generation_log_fields(request_suffix, "mp4_file", model_type, avatar_id=avatar_id or None, has_audio_url=bool(audio_url), use_face_crop=use_face_crop, base_seed=base_seed))
    increment("soulx.generation.requests", tags=tags)
    pipeline = init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)

    media_info = {
        "avatar_id": avatar_id or None,
        "driving": driving.filename if driving else None,
        "audio": audio.filename if audio else None,
        "audio_url": audio_url or None,
    }
    args_dict = {
        "model_type": model_type,
        "base_seed": base_seed,
        "use_face_crop": use_face_crop,
        "silence_padding_sec": silence_padding_sec,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "expression_scale": expression_scale,
    }
    from observability import log_request_summary
    log_request_summary(
        endpoint="/generate_video",
        request_id=request_suffix,
        client_ip=request.client.host if request.client else "unknown",
        media_info=media_info,
        args=args_dict,
        status="STARTED",
    )

    video_path = os.path.join(TEMP_OUTPUT_DIR, f"output_{ts}.mp4")
    try:
        with traced_span("generation.file", request_id=request_suffix, model_type=model_type):
            await generate_mp4(
                pipeline,
                resolved_driving_path,
                temp_audio_path,
                video_path,
                base_seed=base_seed,
                use_face_crop=use_face_crop,
                silence_padding_sec=silence_padding_sec,
                preserve_aspect_ratio=preserve_aspect_ratio,
                output_scale=1,
                encoder_crf=20,
                encoder_preset="veryfast",
                request_id=request_suffix,
                expression_scale=expression_scale,
            )
        duration_ms = (time.time() - request_start) * 1000
        log_request_summary(
            endpoint="/generate_video",
            request_id=request_suffix,
            client_ip=request.client.host if request.client else "unknown",
            media_info=media_info,
            args=args_dict,
            status="COMPLETED",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        log_event("generation_failed", **generation_log_fields(request_suffix, "mp4_file", model_type, error=str(exc), status="failure"))
        increment("soulx.generation.failure", tags=generation_metric_tags("mp4_file", model_type, status="failure", endpoint="generate_video"))
        duration_ms = (time.time() - request_start) * 1000
        log_request_summary(
            endpoint="/generate_video",
            request_id=request_suffix,
            client_ip=request.client.host if request.client else "unknown",
            media_info=media_info,
            args=args_dict,
            status="FAILED",
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise

    duration_ms = round((time.time() - request_start) * 1000, 2)
    log_event("generation_completed", **generation_log_fields(request_suffix, "mp4_file", model_type, duration_ms=duration_ms, output_path=video_path, status="success"))
    increment("soulx.generation.success", tags=generation_metric_tags("mp4_file", model_type, status="success", endpoint="generate_video"))
    distribution("soulx.generation.duration_ms", duration_ms, tags=tags)

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="generated_video.mp4",
    )


@app.post(
    "/pregen/generate",
    summary="Generate complete MP4 file from EFS/URL audio (JSON)",
    description="Generates the full video and returns a downloadable MP4 file. Accepts a JSON body with `avatar_id`, `audio` (bare filename resolved from EFS or HTTP/HTTPS URL), and an optional `args` JSON string that can include `model_type`, `base_seed`, `use_face_crop`, `silence_padding_sec`, `preserve_aspect_ratio`, and `expression_scale`.",
    responses={
        200: {
            "description": "Generated MP4 file.",
            "content": {
                "video/mp4": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters or media input."},
        503: {"model": CapacityErrorResponse, "description": "Server is at concurrent stream capacity."},
    },
)
async def pregen_generate(request: StreamEfsRequest, http_request: Request):
    request_start = time.time()
    if _session_manager is None or _session_manager.is_at_capacity():
        return JSONResponse(
            status_code=503,
            content={
                "error": "CAPACITY_FULL",
                "message": (
                    f"Server is at capacity ({MAX_CONCURRENT_STREAMS} concurrent streams). "
                    "Please try again later."
                ),
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    args = _parse_args(request.args)

    try:
        model_type = normalize_model_type(args.model_type)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # Validate mutually exclusive flags
    if args.use_face_crop and args.preserve_aspect_ratio:
        return JSONResponse(
            status_code=400,
            content={
                "error": "use_face_crop and preserve_aspect_ratio cannot both be true. "
                        "Use use_face_crop=true for face-focused square videos, "
                        "or use_face_crop=false + preserve_aspect_ratio=true for aspect-ratio-preserved videos."
            }
        )

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    ts = int(time.time())
    request_suffix = f"pre_{ts}_{uuid.uuid4().hex[:8]}"

    try:
        temp_audio_path = await download_audio_from_url(request.audio, request_suffix)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    resolved_driving_path = await _resolve_driving_image_path(
        request_suffix, None, request.avatar_id, args.use_face_crop, args.preserve_aspect_ratio
    )
    if isinstance(resolved_driving_path, JSONResponse):
        return resolved_driving_path

    if args.preserve_aspect_ratio:
        logger.debug(
            f"[{request_suffix}] pregen_generate preserve_aspect_ratio={args.preserve_aspect_ratio} "
            f"background_cache_disabled={SOULX_DISABLE_BACKGROUND_CACHE} "
            f"resolved_driving_path={resolved_driving_path}"
        )

    pipeline = init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)
    tags = generation_metric_tags("mp4_file", model_type, endpoint="pregen_generate")
    log_event("generation_started", **generation_log_fields(request_suffix, "mp4_file", model_type, avatar_id=request.avatar_id, has_audio_url=True, use_face_crop=args.use_face_crop, base_seed=args.base_seed))
    increment("soulx.generation.requests", tags=tags)

    media_info = {
        "avatar_id": request.avatar_id or None,
        "audio": request.audio or None,
    }
    args_dict = {
        "model_type": model_type,
        "base_seed": args.base_seed,
        "use_face_crop": args.use_face_crop,
        "silence_padding_sec": args.silence_padding_sec,
        "preserve_aspect_ratio": args.preserve_aspect_ratio,
        "expression_scale": args.expression_scale,
    }
    from observability import log_request_summary
    log_request_summary(
        endpoint="/pregen/generate",
        request_id=request_suffix,
        client_ip=http_request.client.host if http_request.client else "unknown",
        media_info=media_info,
        args=args_dict,
        status="STARTED",
    )

    video_path = os.path.join(TEMP_OUTPUT_DIR, f"output_{ts}.mp4")
    try:
        with traced_span("generation.file", request_id=request_suffix, model_type=model_type):
            await generate_mp4(
                pipeline,
                resolved_driving_path,
                temp_audio_path,
                video_path,
                base_seed=args.base_seed,
                use_face_crop=args.use_face_crop,
                silence_padding_sec=args.silence_padding_sec,
                preserve_aspect_ratio=args.preserve_aspect_ratio,
                request_id=request_suffix,
                expression_scale=args.expression_scale,
            )
        duration_ms = (time.time() - request_start) * 1000
        log_request_summary(
            endpoint="/pregen/generate",
            request_id=request_suffix,
            client_ip=http_request.client.host if http_request.client else "unknown",
            media_info=media_info,
            args=args_dict,
            status="COMPLETED",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        log_event("generation_failed", **generation_log_fields(request_suffix, "mp4_file", model_type, error=str(exc), status="failure"))
        increment("soulx.generation.failure", tags=generation_metric_tags("mp4_file", model_type, status="failure", endpoint="pregen_generate"))
        duration_ms = (time.time() - request_start) * 1000
        log_request_summary(
            endpoint="/pregen/generate",
            request_id=request_suffix,
            client_ip=http_request.client.host if http_request.client else "unknown",
            media_info=media_info,
            args=args_dict,
            status="FAILED",
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise

    duration_ms = round((time.time() - request_start) * 1000, 2)
    log_event("generation_completed", **generation_log_fields(request_suffix, "mp4_file", model_type, duration_ms=duration_ms, output_path=video_path, status="success"))
    increment("soulx.generation.success", tags=generation_metric_tags("mp4_file", model_type, status="success", endpoint="pregen_generate"))
    distribution("soulx.generation.duration_ms", duration_ms, tags=tags)

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="generated_video.mp4",
    )


@app.get("/", summary="Health check", response_model=HealthResponse)
async def root_health():
    """Primary health check endpoint for load balancers and monitoring.
    Returns 503 until model warmup completes successfully."""
    import observability
    if not observability.server_started:
        return JSONResponse(
            {
                "success": False,
                "configured": False,
                "runpodStatus": "warming_up",
                "apiUrl": "local"
            },
            status_code=503,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )
    return {
        "success": True,
        "configured": True,
        "runpodStatus": "running",
        "apiUrl": "local"
    }

@app.get("/admin", summary="Serve frontend admin UI", responses={404: {"model": ErrorResponse, "description": "Static UI not found."}})
async def admin_index():
    static_path = resolve_static_index()
    if static_path and static_path.is_file():
        with open(static_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({"error": "Static UI not found"}, status_code=404)

@app.get("/vite.svg", summary="Serve Vite SVG asset", responses={404: {"model": ErrorResponse, "description": "vite.svg not found."}})
async def vite_svg():
    static_path = resolve_static_index()
    static_dir = static_path.parent if static_path else Path("/opt/deployment/static")
    svg_path = static_dir / "vite.svg"
    if svg_path.is_file():
        return FileResponse(svg_path)
    return JSONResponse({"error": "vite.svg not found"}, status_code=404)

@app.get("/runpod/health", summary="Legacy health check (deprecated)", response_model=HealthResponse, deprecated=True)
@app.get("/api/runpod/health", include_in_schema=False)
async def runpod_health():
    """Legacy health endpoint. Use GET / instead."""
    import observability
    if not observability.server_started:
        return JSONResponse(
            {
                "success": False,
                "configured": False,
                "runpodStatus": "warming_up",
                "apiUrl": "local"
            },
            status_code=503,
        )
    return {
        "success": True,
        "configured": True,
        "runpodStatus": "running",
        "apiUrl": "local"
    }

@app.get("/avatars/list", summary="Get avatar list from avatars.json")
async def get_avatars_list():
    avatars_json_path = Path("/shared_volume/avatar_data/avatars/avatars.json")
    if avatars_json_path.exists():
        with open(avatars_json_path, "r") as f:
            return JSONResponse(await asyncio.get_event_loop().run_in_executor(None, json.load, f))
    return JSONResponse({"error": "avatars.json not found"}, status_code=404)

@app.get("/avatars/{avatar_id}/idle", summary="Get idle video for avatar")
async def get_avatar_idle_video(avatar_id: str):
    """
    Returns the idle video file (mp4 or gif) for the specified avatar.
    Checks for synq_idle.mp4 first, then idle.mp4, and finally idle.gif.
    Returns 404 if none exists.
    """
    avatar_dir = _avatar_image_root / avatar_id
    if not avatar_dir.is_dir():
        logger.warning(f"[get_avatar_idle_video] Avatar folder not found for avatar_id='{avatar_id}': {avatar_dir}")
        return JSONResponse({"error": f"Avatar folder not found for avatar_id '{avatar_id}'"}, status_code=404)
    
    synq_idle_mp4 = avatar_dir / "synq_idle.mp4"
    if synq_idle_mp4.is_file():
        logger.info(f"[get_avatar_idle_video] Serving synq_idle.mp4 for avatar_id='{avatar_id}': {synq_idle_mp4}")
        return FileResponse(synq_idle_mp4, media_type="video/mp4")

    idle_mp4 = avatar_dir / "idle.mp4"
    if idle_mp4.is_file():
        logger.info(f"[get_avatar_idle_video] Serving idle.mp4 for avatar_id='{avatar_id}': {idle_mp4}")
        return FileResponse(idle_mp4, media_type="video/mp4")
    
    idle_gif = avatar_dir / "idle.gif"
    if idle_gif.is_file():
        logger.info(f"[get_avatar_idle_video] Serving idle.gif for avatar_id='{avatar_id}': {idle_gif}")
        return FileResponse(idle_gif, media_type="image/gif")
    
    logger.warning(f"[get_avatar_idle_video] No idle video found for avatar_id='{avatar_id}' in {avatar_dir}")
    return JSONResponse({"error": f"No idle video (synq_idle.mp4, idle.mp4 or idle.gif) found for avatar_id '{avatar_id}'"}, status_code=404)

# Mount static files for avatars after our specific endpoint registrations
# to ensure specific routes (/avatars/list and /avatars/{avatar_id}/idle) match first
# and are not shadowed by the static files wild-card sub-route.
_avatars_dir = Path("/shared_volume/avatar_data/avatars")
if _avatars_dir.exists():
    app.mount("/avatars", StaticFiles(directory=str(_avatars_dir)), name="avatars")

@app.get("/analytics/dashboard", summary="Analytics dashboard summary", response_model=AnalyticsDashboardResponse)
@app.get("/api/analytics/dashboard", include_in_schema=False)
async def analytics_dashboard():
    return {
        "success": True,
        "data": {
            "totalVideos": 0,
            "totalMinutes": 0,
            "averageGenerationTime": 0,
            "successRate": 100,
            "recentActivity": []
        }
    }

def tensor_to_jpeg_base64(tensor_frame):
    """Convert a single frame tensor (H, W, C) to base64-encoded JPEG."""
    frame_np = tensor_frame.numpy().astype(np.uint8)
    img = Image.fromarray(frame_np)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get(
    "/api/status",
    summary="Concurrency status",
    description="Returns current server concurrency capacity and active per-session statistics.",
    response_model=ConcurrencyStatusResponse,
    responses={503: {"model": ErrorResponse, "description": "Server not ready."}},
)
@app.get("/status", include_in_schema=False)
async def stream_status():
    """Return current concurrency capacity and per-session stats."""
    import observability
    if not observability.server_started or _session_manager is None:
        return JSONResponse(
            {"error": "Server not ready", "warmup_complete": observability.server_started},
            status_code=503,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )
    return _session_manager.get_status()

@app.get("/api/compositor/stats", summary="Get aspect ratio compositor cache statistics")
async def compositor_stats():
    """Return cache hit/miss statistics for the aspect ratio compositor."""
    from image_compositor import get_cache_stats
    return get_cache_stats()

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    session = None
    session_id = f"ws_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    send_semaphore = asyncio.Semaphore(1)
    frame_task = None

    async def _send(payload: dict):
        async with send_semaphore:
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception as exc:
                logger.error(f"[{session_id}] send error: {exc}")

    try:
        if _session_manager is None or _session_manager.is_at_capacity():
            await _send({
                'type': 'error',
                'code': 'CAPACITY_FULL',
                'message': (
                    f'Server is at capacity '
                    f'({MAX_CONCURRENT_STREAMS} concurrent streams). '
                    'Please try again later.'
                )
            })
            await websocket.close(code=1013)
            return

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'init':
                logger.info(f"[{session_id}] Initializing streaming session")

                driving_image_b64 = message['driving_image_base64']
                driving_image_bytes = base64.b64decode(driving_image_b64)

                try:
                    img = Image.open(io.BytesIO(driving_image_bytes))
                    width, height = img.size
                    model_side = _model_output_side()
                    if width != model_side or height != model_side:
                        await _send({
                            'type': 'error',
                            'message': (
                                f'Invalid image dimensions: {width}x{height}px. '
                                f'Driving images must be exactly {model_side}x{model_side}px.'
                            )
                        })
                        continue
                    logger.info(f"[{session_id}] Image validation passed: {width}x{height}px")
                except Exception as e:
                    await _send({
                        'type': 'error',
                        'message': f'Invalid image format: {str(e)}. Please upload a valid image file.'
                    })
                    continue

                os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
                temp_driving_path = os.path.join(TEMP_UPLOAD_DIR, f"driving_{session_id}.png")
                with open(temp_driving_path, "wb") as f:
                    f.write(driving_image_bytes)

                model_type = normalize_model_type(message.get('model_type', 'lite'))
                base_seed = message.get('base_seed', 42)
                use_face_crop = message.get('use_face_crop', False)
                jpeg_quality = _validate_jpeg_quality(message.get('jpeg_quality', 85))

                pipeline = init_pipeline(FLASHHEAD_CKPT_DIR, WAV2VEC_DIR, model_type)

                ctx = SessionContext(
                    session_id=session_id,
                    cond_image_path_or_dir=temp_driving_path,
                    base_seed=base_seed,
                    use_face_crop=use_face_crop,
                )

                loop = asyncio.get_event_loop()
                async with _session_manager.gpu_semaphore:
                    await loop.run_in_executor(
                        None,
                        prepare_session_base_data,
                        pipeline,
                        ctx,
                    )

                infer_params = get_infer_params()
                try:
                    session = _session_manager.create_session(
                        session_id=session_id,
                        pipeline=pipeline,
                        ctx=ctx,
                        infer_params=infer_params,
                    )
                except CapacityError as cap_err:
                    await _send({
                        'type': 'error',
                        'code': 'CAPACITY_FULL',
                        'message': str(cap_err),
                    })
                    await websocket.close(code=1013)
                    return

                async def frame_generator():
                    async def on_frame(frame_tensor, frame_index):
                        jpeg_base64 = base64.b64encode(
                            _encode_jpeg(frame_tensor.numpy().astype(np.uint8), jpeg_quality)
                        ).decode("utf-8")
                        await _send({
                            'type': 'frame',
                            'jpeg_base64': jpeg_base64,
                            'frame_index': frame_index,
                            'timestamp': int(time.time() * 1000),
                        })

                    await session.generate_frames(on_frame)

                frame_task = asyncio.create_task(frame_generator())

                await _send({'type': 'ready', 'session_id': session_id})
                logger.info(f"[{session_id}] Session initialized and ready")

            elif message['type'] == 'audio_chunk':
                if session is None:
                    await _send({'type': 'error', 'message': 'Session not initialized'})
                    continue

                audio_bytes = base64.b64decode(message['audio_data'])
                audio_samples = np.frombuffer(audio_bytes, dtype=np.float32)
                session.add_audio_chunk(audio_samples)

                stats = session.get_stats()
                await _send({
                    'type': 'stats',
                    'frames_generated': stats['frames_generated'],
                    'audio_buffered_ms': stats['buffered_ms'],
                })

            elif message['type'] == 'turn_end':
                logger.info(f"[{session_id}] Turn ended, flushing frames")

                if session:
                    async def on_final_frame_turn(frame_tensor, frame_index):
                        jpeg_base64 = base64.b64encode(
                            _encode_jpeg(frame_tensor.numpy().astype(np.uint8), jpeg_quality)
                        ).decode("utf-8")
                        await _send({
                            'type': 'frame',
                            'jpeg_base64': jpeg_base64,
                            'frame_index': frame_index,
                            'timestamp': int(time.time() * 1000),
                        })

                    await session.finalize(on_final_frame_turn)

                await _send({'type': 'stream_complete'})

            elif message['type'] == 'end':
                logger.info(f"[{session_id}] Ending session")

                if session:
                    session.stop()
                    if frame_task and not frame_task.done():
                        frame_task.cancel()
                        try:
                            await frame_task
                        except asyncio.CancelledError:
                            pass

                    async def on_final_frame_end(frame_tensor, frame_index):
                        jpeg_base64 = base64.b64encode(
                            _encode_jpeg(frame_tensor.numpy().astype(np.uint8), jpeg_quality)
                        ).decode("utf-8")
                        await _send({
                            'type': 'frame',
                            'jpeg_base64': jpeg_base64,
                            'frame_index': frame_index,
                            'timestamp': int(time.time() * 1000),
                        })

                    await session.finalize(on_final_frame_end)

                await _send({'type': 'complete'})
                break

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] Error: {str(e)}")
        try:
            await _send({'type': 'error', 'message': str(e)})
        except Exception:
            pass
    finally:
        if session is not None:
            _session_manager.close_session(session_id)
        if frame_task and not frame_task.done():
            frame_task.cancel()
        logger.info(f"[{session_id}] Session closed")


if __name__ == "__main__":
    from config import HOST, PORT
    uvicorn.run(app, host=HOST, port=PORT)
