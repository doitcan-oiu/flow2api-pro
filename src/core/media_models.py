"""Pydantic models for the OpenAI images/videos protocols.

These models cover the native media endpoints:

- ``POST /v1/images/generations``
- ``POST /v1/images/edits``
- ``POST /v1/videos``
- ``POST /v1/videos/{video_id}/remix``

They intentionally allow extra fields so upstream gateways (New API, one-api,
etc.) can keep passing vendor extensions such as ``generationConfig`` without
being rejected. Those extras are still read by ``model_resolver`` to pick the
concrete flow2api model key.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


VIDEO_STATUS_QUEUED = "queued"
VIDEO_STATUS_IN_PROGRESS = "in_progress"
VIDEO_STATUS_COMPLETED = "completed"
VIDEO_STATUS_FAILED = "failed"


class ImageGenerationRequest(BaseModel):
    """``POST /v1/images/generations`` request body."""

    prompt: str
    model: Optional[str] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = None
    quality: Optional[str] = None
    response_format: Optional[Literal["url", "b64_json"]] = None
    output_format: Optional[Literal["png", "jpeg", "webp"]] = None
    background: Optional[str] = None
    style: Optional[str] = None
    moderation: Optional[str] = None
    output_compression: Optional[int] = None
    user: Optional[str] = None
    stream: bool = False
    partial_images: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class ImageEditRequest(BaseModel):
    """JSON variant of ``POST /v1/images/edits``.

    The official endpoint is ``multipart/form-data``; that shape is handled in
    the route layer. This model covers clients that post JSON with base64 or
    URL images instead, which is common for proxy deployments.
    """

    prompt: str
    image: Optional[Any] = None
    mask: Optional[Any] = None
    model: Optional[str] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = None
    quality: Optional[str] = None
    response_format: Optional[Literal["url", "b64_json"]] = None
    output_format: Optional[Literal["png", "jpeg", "webp"]] = None
    background: Optional[str] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class VideoCreateRequest(BaseModel):
    """JSON variant of ``POST /v1/videos``.

    ``input_reference`` accepts a data URL, an http(s) URL or a list of those
    for image-to-video and interpolation models. The official multipart form is
    handled separately in the route layer.
    """

    prompt: str
    model: Optional[str] = None
    seconds: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    input_reference: Optional[Any] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class VideoRemixRequest(BaseModel):
    """``POST /v1/videos/{video_id}/remix`` request body."""

    prompt: str
    model: Optional[str] = None
    seconds: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class VideoError(BaseModel):
    """Error detail attached to a failed video job."""

    code: str
    message: str


class VideoJobResource(BaseModel):
    """``video`` object returned by the videos endpoints."""

    id: str
    object: Literal["video"] = "video"
    model: str
    status: str
    progress: int = 0
    created_at: int
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    size: Optional[str] = None
    seconds: Optional[str] = None
    quality: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    error: Optional[VideoError] = None
    # Download link, present once the job reaches a terminal success state.
    # Not part of the official Sora schema (clients are meant to call
    # /content), but gateways such as New API read it directly.
    url: Optional[str] = None

    model_config = ConfigDict(extra="allow")


def build_video_list_payload(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap video resources in the OpenAI list envelope."""
    return {
        "object": "list",
        "data": items,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
        "has_more": False,
    }
