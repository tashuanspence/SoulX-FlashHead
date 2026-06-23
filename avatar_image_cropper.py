import hashlib
import time
from pathlib import Path

from PIL import Image
from loguru import logger

from config import SOULX_DISABLE_AVATAR_CROP_CACHE


class AvatarImageCropper:
    def __init__(self, cache_dir: str, target_size: int = 512):
        self.cache_dir = Path(cache_dir)
        self.target_size = target_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_target_image(self, source_path: str, cache_key: str) -> tuple[str, bool, bool]:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")

        stat_result = source.stat()

        with Image.open(source) as img:
            width, height = img.size

        if width == self.target_size and height == self.target_size:
            logger.info(
                f"Avatar crop service: using original mapped image without crop for cache_key={cache_key} at {source} ({width}x{height})"
            )
            return str(source), False, False

        if SOULX_DISABLE_AVATAR_CROP_CACHE:
            logger.warning(
                f"Avatar crop service: cache disabled via SOULX_DISABLE_AVATAR_CROP_CACHE; "
                f"processing source={source} cache_key={cache_key}"
            )
            return self._crop_and_save(source, cache_key, stat_result, use_cache=False)

        cache_name = self._build_cache_name(source, cache_key, stat_result)
        cached_path = self.cache_dir / cache_name
        if cached_path.is_file():
            logger.info(
                f"Avatar crop service: cache hit for cache_key={cache_key}; using cached center crop {cached_path} from source {source} ({width}x{height})"
            )
            return str(cached_path), True, True

        return self._crop_and_save(source, cache_key, stat_result, use_cache=True)

    def _crop_and_save(
        self,
        source: Path,
        cache_key: str,
        stat_result,
        use_cache: bool,
    ) -> tuple[str, bool, bool]:
        with Image.open(source) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            crop_size = min(img.width, img.height)
            left = max((img.width - crop_size) // 2, 0)
            top = max((img.height - crop_size) // 2, 0)
            right = left + crop_size
            bottom = top + crop_size

            cropped = img.crop((left, top, right, bottom)).resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS,
            )
            save_image = cropped.convert("RGB") if cropped.mode == "RGBA" else cropped
            if use_cache:
                cached_path = self.cache_dir / self._build_cache_name(source, cache_key, stat_result)
            else:
                cached_path = self.cache_dir / (
                    f"{source.stem}_{hashlib.sha1(f'{cache_key}|{source.resolve()}|{time.time_ns()}'.encode('utf-8')).hexdigest()[:12]}.png"
                )
            save_image.save(cached_path, format="PNG")

        logger.info(
            f"Avatar crop service: created {'cached' if use_cache else 'uncached'} center crop for cache_key={cache_key}; source={source} original={width}x{height} mtime_ns={stat_result.st_mtime_ns} size={stat_result.st_size} path={cached_path} target={self.target_size}x{self.target_size}"
        )

        return str(cached_path), True, False

    def _build_cache_name(self, source: Path, cache_key: str, stat_result) -> str:
        suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg", ".jpeg"} else ".png"
        fingerprint = hashlib.sha1(
            f"{cache_key}|{source.resolve()}|{stat_result.st_mtime_ns}|{stat_result.st_size}".encode("utf-8")
        ).hexdigest()[:12]
        return f"{source.stem}_{fingerprint}{suffix if suffix == '.png' else '.png'}"
