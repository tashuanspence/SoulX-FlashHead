import os


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Maximum number of concurrent inference streams.
# Model_Lite on a single RTX4090 supports up to 3 real-time (25+ FPS) streams.
# Raise this value on EC2 p5 (H100) if benchmarks confirm headroom.
MAX_CONCURRENT_STREAMS: int = int(os.environ.get("MAX_CONCURRENT_STREAMS", 3))

# Model checkpoint paths
_EFS_MODEL_DIR = "/shared_volume/forward/soulx-head/models"
_DEFAULT_MODEL_DIR = _EFS_MODEL_DIR if os.path.isdir(_EFS_MODEL_DIR) or os.path.isdir("/shared_volume") else "models"

FLASHHEAD_CKPT_DIR: str = os.environ.get(
    "FLASHHEAD_CKPT_DIR", os.path.join(_DEFAULT_MODEL_DIR, "SoulX-FlashHead-1_3B")
)
WAV2VEC_DIR: str = os.environ.get(
    "WAV2VEC_DIR", os.path.join(_DEFAULT_MODEL_DIR, "wav2vec2-base-960h")
)

# Default model type: "lite" for concurrent streaming, "pro" for highest quality
MODEL_TYPE: str = os.environ.get("MODEL_TYPE", "lite")

# Model types to fully load and compile during startup warmup.
# Defaults to [MODEL_TYPE] to preserve current behavior. Set to "lite,pro" to
# pre-warm both variants so the first /stream-efs request for either type is fast.
_raw_warmup = os.environ.get("SOULX_WARMUP_MODEL_TYPES", MODEL_TYPE)
WARMUP_MODEL_TYPES: list[str] = [
    mt.strip() for mt in _raw_warmup.split(",") if mt.strip()
]

# Distributed / multi-GPU (USP) settings
WORLD_SIZE: int = int(os.environ.get("WORLD_SIZE", 1))
RANK: int = int(os.environ.get("RANK", 0))

# Server settings
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", 8000))

# Temp directories
TEMP_UPLOAD_DIR: str = os.environ.get("TEMP_UPLOAD_DIR", "temp_uploads")
TEMP_OUTPUT_DIR: str = os.environ.get("TEMP_OUTPUT_DIR", "temp_outputs")
TTS_AUDIO_DIR: str = os.environ.get("TTS_AUDIO_DIR", "/shared_volume/avatar_data/tts")
SOULX_AVATAR_IMAGE_ROOT: str = os.environ.get("SOULX_AVATAR_IMAGE_ROOT", "/shared_volume/avatar_data/avatars")
SOULX_AVATAR_CROP_CACHE_DIR: str = os.environ.get(
    "SOULX_AVATAR_CROP_CACHE_DIR",
    os.path.join(SOULX_AVATAR_IMAGE_ROOT, ".cache", "cropped")
)
SOULX_DISABLE_BACKGROUND_CACHE: bool = _env_flag("SOULX_DISABLE_BACKGROUND_CACHE", False)
SOULX_DISABLE_AVATAR_CROP_CACHE: bool = _env_flag("SOULX_DISABLE_AVATAR_CROP_CACHE", False)

# MPEG stream fragment duration in microseconds (1,000 µs = 1 ms).
# Default 250 000 µs = 250 ms (~6 frames at 25fps).  Configurable via env var (in ms).
# Lower values = smoother real-time playback but more overhead.
# 250ms balances real-time performance with encoding efficiency for VMs.
_frag_duration_ms: int = int(os.environ.get("FRAGMENT_DURATION_MS", 100))
FRAGMENT_DURATION_US: int = _frag_duration_ms * 1000

# Default encoder quality settings (overridable per-request)
SOULX_ENCODER_CRF: int = int(os.environ.get("SOULX_ENCODER_CRF", 20))
SOULX_ENCODER_PRESET: str = os.environ.get("SOULX_ENCODER_PRESET", "veryfast")
