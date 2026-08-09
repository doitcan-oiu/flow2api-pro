"""Shared helpers for the native media endpoints (`/v1/images`, `/v1/videos`).

This module keeps the pieces that both media routers need in one place:

- resolving an incoming model name to a concrete ``MODEL_CONFIG`` key
- turning request parameters (``size`` / ``quality`` / ``seconds``) into the
  ``generationConfig`` shape ``model_resolver`` already understands
- loading reference images from data URLs, http(s) URLs or uploaded files
- running ``GenerationHandler`` non-streaming and extracting the media URL

The routers stay thin and protocol-specific, the generation pipeline stays
untouched.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import re

from fastapi import HTTPException, Request, UploadFile

from ..core.logger import debug_logger
from ..core.model_resolver import resolve_model_name
from ..services.generation_handler import MODEL_CONFIG, GenerationHandler

MARKDOWN_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
HTML_VIDEO_RE = re.compile(r"<video[^>]+src=['\"](.*?)['\"]", re.IGNORECASE)
DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
SECONDS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_VIDEO_MODEL = "veo_3_1_t2v_fast"

# Durations that map to a flow2api model suffix.
SUPPORTED_VIDEO_SECONDS = ("4", "6", "8")

_generation_handler: Optional[GenerationHandler] = None


def set_generation_handler(handler: GenerationHandler) -> None:
    """Inject the shared generation handler instance."""
    global _generation_handler
    _generation_handler = handler


def ensure_generation_handler() -> GenerationHandler:
    if _generation_handler is None:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")
    return _generation_handler


def get_generation_handler() -> Optional[GenerationHandler]:
    return _generation_handler


@dataclass
class MediaParams:
    """Normalized generation parameters shared by images and videos."""

    model: str
    prompt: str
    images: List[bytes] = field(default_factory=list)
    size: Optional[str] = None
    seconds: Optional[str] = None
    quality: Optional[str] = None


class _ParamCarrier:
    """Minimal object exposing ``generationConfig`` for ``model_resolver``.

    ``resolve_model_name`` reads ``generationConfig`` (and Pydantic extras) to
    infer aspect ratio and resolution. Building a tiny carrier keeps the
    resolver untouched while letting the media endpoints reuse it.
    """

    def __init__(self, generation_config: Dict[str, Any], extra: Dict[str, Any]):
        self.generationConfig = generation_config
        self.__pydantic_extra__ = extra


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_seconds(value: Any) -> Optional[str]:
    """Normalize ``seconds`` into one of the supported duration buckets."""
    raw = _clean_str(value)
    if not raw:
        return None

    match = SECONDS_RE.match(raw)
    if not match:
        return None

    try:
        requested = float(match.group(1))
    except ValueError:
        return None
    if requested <= 0:
        return None

    # Snap to the nearest supported duration so `seconds=5` still works.
    nearest = min(SUPPORTED_VIDEO_SECONDS, key=lambda item: abs(requested - float(item)))
    if float(nearest) != requested:
        debug_logger.log_info(
            f"[MEDIA] seconds={raw} 不受支持，已就近映射到 {nearest}s"
        )
    return nearest


def _extra_params(raw_request_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw_request_body, dict):
        return {}
    extra: Dict[str, Any] = {}
    for key in ("generationConfig", "generation_config", "extra_body", "extraBody"):
        if key in raw_request_body:
            extra[key] = raw_request_body[key]
    return extra


def resolve_media_model(
    *,
    requested_model: Optional[str],
    default_model: str,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    seconds: Optional[str] = None,
    images: Optional[List[bytes]] = None,
    passthrough: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve a public model name into an internal ``MODEL_CONFIG`` key."""
    model = _clean_str(requested_model) or default_model

    normalized_seconds = normalize_seconds(seconds)
    if normalized_seconds:
        candidate = f"{model}_{normalized_seconds}s"
        # Only apply the duration suffix when it names a real variant.
        if candidate in MODEL_CONFIG or _is_known_alias(candidate):
            model = candidate
        else:
            debug_logger.log_info(
                f"[MEDIA] 模型 {model} 无 {normalized_seconds}s 变体，忽略 seconds"
            )

    generation_config: Dict[str, Any] = {}
    if size:
        generation_config["size"] = size
    if quality:
        generation_config["quality"] = quality

    carrier = _ParamCarrier(generation_config, _extra_params(passthrough))
    resolved = resolve_model_name(
        model=model,
        request=carrier,
        model_config=MODEL_CONFIG,
        images=images,
    )

    if resolved not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    if resolved != model:
        debug_logger.log_info(f"[MEDIA] 模型名已转换: {model} → {resolved}")
    return resolved


def _is_known_alias(candidate: str) -> bool:
    from ..core.model_resolver import IMAGE_BASE_MODELS, VIDEO_BASE_MODELS

    return candidate in VIDEO_BASE_MODELS or candidate in IMAGE_BASE_MODELS


def assert_model_type(model: str, expected: str) -> Dict[str, Any]:
    """Ensure the resolved model matches the endpoint's media type."""
    model_config = MODEL_CONFIG.get(model)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    actual = model_config.get("type")
    if actual != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' is a {actual} model and cannot be used on the "
                f"{expected} endpoint"
            ),
        )
    return model_config


def decode_data_url(data_url: str) -> Tuple[str, bytes]:
    match = DATA_URL_RE.match(data_url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid data URL")
    try:
        return match.group("mime"), base64.b64decode(match.group("data"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}")


async def load_image_bytes(value: Any) -> List[bytes]:
    """Load one or more reference images from mixed client input."""
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        images: List[bytes] = []
        for item in value:
            images.extend(await load_image_bytes(item))
        return images

    if isinstance(value, (bytes, bytearray)):
        return [bytes(value)] if value else []

    if isinstance(value, dict):
        # Support {"type": "image_url", "image_url": {"url": ...}} and similar.
        for key in ("url", "image_url", "data", "b64_json", "fileUri", "file_uri"):
            if key in value:
                nested = value[key]
                if isinstance(nested, dict):
                    nested = nested.get("url") or nested.get("data")
                return await load_image_bytes(nested)
        raise HTTPException(status_code=400, detail="Unsupported image reference object")

    text = _clean_str(value)
    if not text:
        return []

    if text.startswith("data:"):
        _, image_bytes = decode_data_url(text)
        return [image_bytes]

    if text.startswith("http://") or text.startswith("https://") or "/tmp/" in text:
        from .routes import retrieve_image_data

        image_bytes = await retrieve_image_data(text)
        if not image_bytes:
            raise HTTPException(status_code=400, detail=f"Failed to load image from {text}")
        return [image_bytes]

    # Bare base64 payload without the data URL prefix.
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported image reference")
    if not decoded:
        raise HTTPException(status_code=400, detail="Empty image reference")
    return [decoded]


async def read_upload_files(uploads: Optional[List[UploadFile]]) -> List[bytes]:
    """Read uploaded multipart files into raw bytes."""
    images: List[bytes] = []
    for upload in uploads or []:
        if upload is None:
            continue
        content = await upload.read()
        if content:
            images.append(content)
    return images


def get_request_base_url(request: Request) -> Optional[str]:
    """Derive the externally reachable base URL from request headers."""
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    forwarded_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    host = (forwarded_host or request.headers.get("host") or "").strip()
    if not host:
        return None
    proto = forwarded_proto or request.url.scheme or "http"
    return f"{proto}://{host}"


def parse_handler_result(result: str) -> Dict[str, Any]:
    try:
        return json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return {"result": result}


def handler_error_status(payload: Dict[str, Any]) -> int:
    error = payload.get("error")
    if isinstance(error, dict):
        status_code = error.get("status_code")
        if isinstance(status_code, int):
            return status_code
        if isinstance(status_code, str) and status_code.isdigit():
            return int(status_code)
        return 400
    return 200


def handler_error_message(payload: Dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "Generation failed"


def build_error_payload(status_code: int, message: str) -> Dict[str, Any]:
    """Build an OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": "server_error" if status_code >= 500 else "invalid_request_error",
            "code": None,
            "param": None,
        }
    }


def extract_media_url(payload: Dict[str, Any]) -> Optional[str]:
    """Pull the generated asset URL out of a handler payload."""
    direct_url = payload.get("url")
    if isinstance(direct_url, str) and direct_url.strip():
        return direct_url.strip()

    choices = payload.get("choices") or []
    content = ""
    if choices:
        message = choices[0].get("message") or {}
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
    if not content:
        content = payload.get("result") if isinstance(payload.get("result"), str) else ""
    content = (content or "").strip()
    if not content:
        return None

    image_match = MARKDOWN_IMAGE_RE.search(content)
    if image_match:
        return image_match.group(1).strip()

    video_match = HTML_VIDEO_RE.search(content)
    if video_match:
        return video_match.group(1).strip()

    if content.startswith("http://") or content.startswith("https://"):
        return content
    return None


async def run_generation(
    params: MediaParams,
    *,
    base_url_override: Optional[str] = None,
) -> str:
    """Run the generation pipeline non-streaming and return the media URL."""
    handler = ensure_generation_handler()

    result: Optional[str] = None
    async for chunk in handler.handle_generation(
        model=params.model,
        prompt=params.prompt,
        images=params.images or None,
        stream=False,
        base_url_override=base_url_override,
    ):
        result = chunk

    if result is None:
        raise HTTPException(status_code=500, detail="Generation failed: no response")

    payload = parse_handler_result(result)
    if "error" in payload:
        raise HTTPException(
            status_code=handler_error_status(payload),
            detail=handler_error_message(payload),
        )

    url = extract_media_url(payload)
    if not url:
        raise HTTPException(status_code=500, detail="Generation failed: no media returned")
    return url


async def read_media_bytes(url: str) -> Optional[bytes]:
    """Read generated media bytes, preferring the local cache."""
    from urllib.parse import urlparse

    handler = get_generation_handler()
    file_cache = getattr(handler, "file_cache", None)

    if file_cache is not None and "/tmp/" in url:
        try:
            filename = urlparse(url).path.split("/tmp/")[-1]
            path = file_cache.get_cache_path(filename)
            if path.exists() and path.is_file():
                data = path.read_bytes()
                if data:
                    return data
        except Exception as exc:
            debug_logger.log_warning(f"[MEDIA] 本地缓存读取失败: {exc}")

    from .routes import retrieve_image_data

    return await retrieve_image_data(url)
