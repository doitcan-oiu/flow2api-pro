"""Automatic video model routing based on reference image count.

Flow exposes three separate model families for what users think of as one
task, and the only difference is how many reference images they accept:

===================  ============  ===========================================
image count          video_type    example
===================  ============  ===========================================
0                    ``t2v``       ``veo_3_1_t2v_lite_8s_landscape``
1 (first frame)      ``i2v``       ``veo_3_1_i2v_lite_8s_landscape``
2 (first + last)     ``i2v``       ``veo_3_1_interpolation_lite_8s_landscape``
up to 3 (multi-ref)  ``r2v``       ``veo_3_1_r2v_fast_8s``
===================  ============  ===========================================

Asking callers to pick the right family by hand is a poor interface: sending
two images to a first-frame-only model is a hard 400, even though a sibling
model in the same family handles exactly that case. This module picks the
sibling that matches the supplied image count.

Routing preserves what the caller actually expressed - duration, orientation
and quality tier - and only swaps the part they got wrong. If nothing fits the
image count the caller keeps their original model and the normal validation
error is raised, so the behaviour stays explicit rather than guessing.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .logger import debug_logger

# Tokens describing a model's *role* (which family it belongs to). These are
# the parts we are allowed to change when routing.
_ROLE_TOKENS = frozenset({"t2v", "i2v", "r2v", "interpolation", "s", "fl"})

# Tokens describing orientation, compared separately so it is never traded
# away for a better tier match.
_GEO_TOKENS = frozenset({"landscape", "portrait"})

_DURATION_RE = re.compile(r"^\d+s$")

_MODEL_PREFIX = "veo_3_1_"

# `extend` continues an existing video and is driven by `video_media_id`
# rather than reference images, so it must never be auto-selected.
_NON_ROUTABLE_VIDEO_TYPES = frozenset({"extend"})

# Upper bound when probing which image counts a family supports. Flow's
# multi-reference models cap at 3, so there is nothing to find beyond that.
_MAX_PROBE_IMAGES = 3


def _tokenize(model_name: str) -> List[str]:
    trimmed = model_name[len(_MODEL_PREFIX):] if model_name.startswith(_MODEL_PREFIX) else model_name
    return [token for token in trimmed.split("_") if token]


def _profile(model_name: str, model_config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], frozenset]:
    """Describe a model as (duration, aspect_ratio, quality/tier tokens)."""
    tokens = _tokenize(model_name)
    duration = next((token for token in tokens if _DURATION_RE.match(token)), None)
    tier = frozenset(
        token
        for token in tokens
        if token not in _ROLE_TOKENS
        and token not in _GEO_TOKENS
        and not _DURATION_RE.match(token)
    )
    return duration, model_config.get("aspect_ratio"), tier


def accepts_image_count(model_config: Dict[str, Any], image_count: int) -> bool:
    """Report whether a model can handle exactly ``image_count`` images."""
    if model_config.get("type") != "video":
        return False

    video_type = model_config.get("video_type")
    if video_type in _NON_ROUTABLE_VIDEO_TYPES:
        return False

    if image_count <= 0:
        # Text-to-video is the only family that runs without references.
        return video_type == "t2v"

    if not model_config.get("supports_images"):
        return False

    max_images = model_config.get("max_images")
    if not isinstance(max_images, int):
        return False

    min_images = model_config.get("min_images") or 0
    return min_images <= image_count <= max_images


def describe_image_range(min_images: int, max_images: Optional[int]) -> str:
    """Render an image-count requirement as ``1``, ``1-2`` or ``up to 3``."""
    if max_images is None:
        return f"at least {min_images}"
    if min_images == max_images:
        return str(max_images)
    if min_images <= 0:
        return f"up to {max_images}"
    return f"{min_images}-{max_images}"


def supported_image_counts(
    model: str,
    model_config_map: Dict[str, Dict[str, Any]],
) -> List[int]:
    """List the image counts reachable from ``model``'s family.

    Used to tell the client which counts would work when the requested one
    has no matching sibling.
    """
    current = model_config_map.get(model)
    if not current or current.get("type") != "video":
        return []
    if current.get("video_type") in _NON_ROUTABLE_VIDEO_TYPES:
        return []

    duration, aspect_ratio, tier = _profile(model, current)
    counts = set()

    for name, config in model_config_map.items():
        candidate_duration, candidate_aspect, candidate_tier = _profile(name, config)
        # Only consider the same duration/orientation/quality family.
        if candidate_duration != duration or candidate_aspect != aspect_ratio:
            continue
        if tier != candidate_tier:
            continue
        for count in range(0, _MAX_PROBE_IMAGES + 1):
            if accepts_image_count(config, count):
                counts.add(count)

    return sorted(counts)


def route_video_model(
    model: str,
    image_count: int,
    model_config_map: Dict[str, Dict[str, Any]],
) -> str:
    """Return the sibling video model matching ``image_count``.

    The original ``model`` is returned unchanged when it already accepts the
    given image count, when it is not a video model, or when no sibling fits.
    """
    current = model_config_map.get(model)
    if not current or current.get("type") != "video":
        return model

    if current.get("video_type") in _NON_ROUTABLE_VIDEO_TYPES:
        return model

    if accepts_image_count(current, image_count):
        return model

    duration, aspect_ratio, tier = _profile(model, current)

    best_name: Optional[str] = None
    best_score: Optional[Tuple[bool, bool, int]] = None

    for name, config in model_config_map.items():
        if not accepts_image_count(config, image_count):
            continue

        candidate_duration, candidate_aspect, candidate_tier = _profile(name, config)
        score = (
            candidate_duration == duration,
            candidate_aspect == aspect_ratio,
            # Reward shared quality tokens, penalise ones either side lacks,
            # so `lite` stays with `lite` and `ultra` stays with `ultra`.
            len(tier & candidate_tier) - len(tier ^ candidate_tier),
        )
        if best_score is None or score > best_score:
            best_name, best_score = name, score

    if best_name is None or best_name == model:
        return model

    debug_logger.log_info(
        f"[VIDEO_ROUTING] 参考图数量={image_count}，模型自动切换: {model} → {best_name}"
    )
    return best_name
