"""
Image Compositor for Aspect Ratio Preservation
-----------------------------------------------
Handles background preparation and frame compositing to preserve non-square
aspect ratios (16:9, 9:16, etc.) in generated videos.

Features LRU caching for background preparation to optimize repeated requests
with the same driving image (common in chat applications).
"""

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

from config import SOULX_DISABLE_BACKGROUND_CACHE

# Cache statistics
_cache_hits = 0
_cache_misses = 0

# Aspect ratio tolerance for determining if an image is "square enough"
SQUARE_TOLERANCE = 0.05  # 5%


@lru_cache(maxsize=50)
def _prepare_background_cached(
    image_path: str, mtime_ns: int, file_size: int, preserve_aspect_ratio: bool,
    model_output_size: int = 512,
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Internal cached function for background preparation.
    
    Args:
        image_path: Path to the driving image
        mtime_ns: File modification time in nanoseconds (for cache invalidation)
        file_size: File size in bytes (for cache invalidation)
        preserve_aspect_ratio: Whether to preserve aspect ratio
        model_output_size: Side length of the square model output frame (512 * output_scale)

    Returns:
        Tuple of (background_array, x_offset, y_offset, output_width, output_height)
    """
    global _cache_misses
    _cache_misses += 1
    
    logger.info(
        f"[image_compositor] Cache miss - preparing background for {image_path} "
        f"(mtime_ns={mtime_ns}, size={file_size}, model_output_size={model_output_size})"
    )
    
    # Load the original image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    # Determine scaling strategy based on aspect ratio
    aspect_ratio = orig_w / orig_h
    
    if aspect_ratio > 1.0:
        # Landscape: scale so height matches model output size
        target_h = model_output_size
        target_w = int(orig_w * (model_output_size / orig_h))
    else:
        # Portrait or square: scale so width matches model output size
        target_w = model_output_size
        target_h = int(orig_h * (model_output_size / orig_w))
    
    # Resize the image
    img_resized = img.resize((target_w, target_h), Image.LANCZOS)
    
    # Convert to numpy array
    background_array = np.array(img_resized, dtype=np.uint8)
    
    # Calculate offsets for center placement of the model output frame
    x_offset = (target_w - model_output_size) // 2 if target_w > model_output_size else 0
    y_offset = (target_h - model_output_size) // 2 if target_h > model_output_size else 0

    logger.info(
        f"[image_compositor] Background prepared: {orig_w}x{orig_h} → {target_w}x{target_h}, "
        f"offsets=({x_offset}, {y_offset}), model_output_size={model_output_size}"
    )
    
    return background_array, x_offset, y_offset, target_w, target_h


def prepare_background(
    image_path: str,
    model_output_size: int = 512,
) -> Optional[Tuple[np.ndarray, int, int, int, int]]:
    """
    Prepare background for aspect ratio preservation with LRU caching.
    
    Args:
        image_path: Path to the driving image
        model_output_size: Side length of the square model output frame (512 * output_scale)

    Returns:
        Tuple of (background_array, x_offset, y_offset, output_width, output_height)
        or None if compositing is not needed
    """
    global _cache_hits
    
    if not os.path.exists(image_path):
        logger.warning(f"[image_compositor] Image not found: {image_path}")
        return None

    if SOULX_DISABLE_BACKGROUND_CACHE:
        logger.warning(
            f"[image_compositor] Background cache disabled via SOULX_DISABLE_BACKGROUND_CACHE; "
            f"preparing uncached background for {image_path}"
        )
        stat_result = os.stat(image_path)
        return _prepare_background_direct(image_path, stat_result, model_output_size)
    
    stat_result = os.stat(image_path)
    mtime_ns = stat_result.st_mtime_ns
    file_size = stat_result.st_size
    
    # Check if we've seen this exact image before (cache hit tracking)
    cache_info_before = _prepare_background_cached.cache_info()
    
    logger.debug(
        f"[image_compositor] Preparing background with cache key: "
        f"path={image_path}, mtime_ns={mtime_ns}, size={file_size}, model_output_size={model_output_size}"
    )
    result = _prepare_background_cached(image_path, mtime_ns, file_size, True, model_output_size)
    
    # Track cache hits
    cache_info_after = _prepare_background_cached.cache_info()
    if cache_info_after.hits > cache_info_before.hits:
        _cache_hits += 1
        logger.debug(f"[image_compositor] Cache hit for {image_path}")
    
    return result


def _prepare_background_direct(
    image_path: str,
    stat_result: os.stat_result,
    model_output_size: int = 512,
) -> Tuple[np.ndarray, int, int, int, int]:
    """Prepare a background without using the LRU cache."""
    global _cache_misses
    _cache_misses += 1

    logger.info(
        f"[image_compositor] Cache bypass - preparing background for {image_path} "
        f"(mtime_ns={stat_result.st_mtime_ns}, size={stat_result.st_size}, model_output_size={model_output_size})"
    )

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    aspect_ratio = orig_w / orig_h

    if aspect_ratio > 1.0:
        target_h = model_output_size
        target_w = int(orig_w * (model_output_size / orig_h))
    else:
        target_w = model_output_size
        target_h = int(orig_h * (model_output_size / orig_w))

    img_resized = img.resize((target_w, target_h), Image.LANCZOS)
    background_array = np.array(img_resized, dtype=np.uint8)

    x_offset = (target_w - model_output_size) // 2 if target_w > model_output_size else 0
    y_offset = (target_h - model_output_size) // 2 if target_h > model_output_size else 0

    logger.info(
        f"[image_compositor] Background prepared without cache: {orig_w}x{orig_h} → {target_w}x{target_h}, "
        f"offsets=({x_offset}, {y_offset}), model_output_size={model_output_size}"
    )

    return background_array, x_offset, y_offset, target_w, target_h


def should_composite(image_path: str, preserve_aspect_ratio: bool) -> bool:
    """
    Determine if compositing should be applied.
    
    Args:
        image_path: Path to the driving image
        preserve_aspect_ratio: User preference for aspect ratio preservation
        
    Returns:
        True if compositing should be applied, False otherwise
    """
    if not preserve_aspect_ratio:
        return False
    
    if not os.path.exists(image_path):
        return False
    
    # Check aspect ratio
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height
    
    # If image is square (within tolerance), no compositing needed
    if abs(aspect_ratio - 1.0) < SQUARE_TOLERANCE:
        logger.info(
            f"[image_compositor] Image is square (aspect={aspect_ratio:.3f}), "
            "skipping compositing"
        )
        return False
    
    return True


def composite_frame(
    frame: np.ndarray,
    background: np.ndarray,
    x_offset: int,
    y_offset: int,
    output_buffer: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Composite a generated frame onto the background.

    Args:
        frame: Generated frame (HxWx3) — any square size
        background: Background array (HxWx3)
        x_offset: X offset for frame placement
        y_offset: Y offset for frame placement
        output_buffer: Pre-allocated output buffer (optional, for performance)

    Returns:
        Composited frame (HxWx3)
    """
    if output_buffer is None:
        output_buffer = np.empty_like(background)
    
    # Copy background into output buffer
    np.copyto(output_buffer, background)
    
    # Paste the frame at the center position using its actual dimensions
    fh, fw = frame.shape[:2]
    output_buffer[y_offset : y_offset + fh, x_offset : x_offset + fw] = frame
    
    return output_buffer


def scale_frame(frame: np.ndarray, output_scale: int) -> np.ndarray:
    """
    Upscale a frame by output_scale using Lanczos interpolation.

    Args:
        frame: Input frame (HxWx3)
        output_scale: Integer multiplier (1=no-op, 2=2x, etc.)

    Returns:
        Upscaled frame as uint8 numpy array
    """
    if output_scale == 1:
        return frame
    h, w = frame.shape[:2]
    return np.array(
        Image.fromarray(frame).resize((w * output_scale, h * output_scale), Image.LANCZOS),
        dtype=np.uint8,
    )


def get_cache_stats() -> dict:
    """
    Get cache statistics for monitoring.
    
    Returns:
        Dict with cache_hits, cache_misses, hit_rate, and cache_size
    """
    cache_info = _prepare_background_cached.cache_info()
    total_requests = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total_requests if total_requests > 0 else 0.0
    
    return {
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate": round(hit_rate, 3),
        "cache_size": cache_info.currsize,
        "cache_maxsize": cache_info.maxsize,
    }


def clear_cache() -> None:
    """Clear the background preparation cache."""
    global _cache_hits, _cache_misses
    _prepare_background_cached.cache_clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("[image_compositor] Cache cleared")
