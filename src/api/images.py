"""Native OpenAI images protocol.

Endpoints:

- ``POST /v1/images/generations`` - text to image
- ``POST /v1/images/edits`` - image to image (JSON or multipart/form-data)

Both return the standard ``{"created": ..., "data": [{"url"|"b64_json": ...}]}``
envelope. ``response_format`` defaults to ``url`` because the proxy already
serves cached assets over HTTP; clients that need inline bytes can request
``b64_json``.
"""

import asyncio
import base64
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..core.auth import verify_api_key_flexible
from ..core.logger import debug_logger
from ..core.media_models import ImageEditRequest, ImageGenerationRequest
from .media_common import (
    DEFAULT_IMAGE_MODEL,
    MediaParams,
    assert_model_type,
    build_error_payload,
    get_request_base_url,
    load_image_bytes,
    read_media_bytes,
    read_upload_files,
    resolve_media_model,
    run_generation,
)

router = APIRouter(prefix="/v1/images", tags=["images"])

MAX_IMAGES = 4


async def _read_json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


async def _build_image_data_entry(
    url: str,
    response_format: str,
) -> Dict[str, Any]:
    """Render one image result in the requested response format."""
    if response_format != "b64_json":
        return {"url": url, "b64_json": None, "revised_prompt": None}

    content = await read_media_bytes(url)
    if not content:
        raise HTTPException(status_code=502, detail="Failed to read generated image")
    return {
        "url": None,
        "b64_json": base64.b64encode(content).decode("ascii"),
        "revised_prompt": None,
    }


async def _generate_images(
    params: MediaParams,
    *,
    count: int,
    response_format: str,
    base_url: Optional[str],
) -> List[Dict[str, Any]]:
    """Run ``count`` generations concurrently and collect the results."""
    tasks = [
        asyncio.create_task(run_generation(params, base_url_override=base_url))
        for _ in range(count)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    urls: List[str] = []
    first_error: Optional[BaseException] = None
    for item in results:
        if isinstance(item, BaseException):
            first_error = first_error or item
            continue
        urls.append(item)

    if not urls:
        if isinstance(first_error, HTTPException):
            raise first_error
        raise HTTPException(
            status_code=500,
            detail=str(first_error) if first_error else "Image generation failed",
        )

    if first_error is not None:
        debug_logger.log_warning(
            f"[IMAGES] 部分生成失败，成功 {len(urls)}/{count}: {first_error}"
        )

    return [await _build_image_data_entry(url, response_format) for url in urls]


def _build_response(
    data: List[Dict[str, Any]],
    *,
    model: str,
    size: Optional[str],
    quality: Optional[str],
    output_format: Optional[str],
    background: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "created": int(time.time()),
        "data": data,
    }
    if size:
        payload["size"] = size
    if quality:
        payload["quality"] = quality
    if output_format:
        payload["output_format"] = output_format
    if background:
        payload["background"] = background
    # Report the resolved public model id, not the upstream internal codename.
    payload["model"] = model
    return payload


@router.post("/generations")
async def create_image(
    request: ImageGenerationRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key_flexible),
):
    """Text-to-image generation."""
    try:
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported on /v1/images/generations",
            )

        passthrough = await _read_json_body(raw_request)
        model = resolve_media_model(
            requested_model=request.model,
            default_model=DEFAULT_IMAGE_MODEL,
            size=request.size,
            quality=request.quality,
            passthrough=passthrough,
        )
        assert_model_type(model, "image")

        count = min(max(request.n or 1, 1), MAX_IMAGES)
        response_format = request.response_format or "url"
        params = MediaParams(
            model=model,
            prompt=prompt,
            size=request.size,
            quality=request.quality,
        )

        data = await _generate_images(
            params,
            count=count,
            response_format=response_format,
            base_url=get_request_base_url(raw_request),
        )
        return JSONResponse(
            content=_build_response(
                data,
                model=model,
                size=request.size,
                quality=request.quality,
                output_format=request.output_format,
                background=request.background,
            )
        )

    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.status_code, str(exc.detail)),
        )
    except Exception as exc:
        debug_logger.log_error(f"[IMAGES] 生成异常: {exc}")
        return JSONResponse(status_code=500, content=build_error_payload(500, str(exc)))


@router.post("/edits")
async def edit_image(
    raw_request: Request,
    api_key: str = Depends(verify_api_key_flexible),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form(None),
    image: Optional[List[UploadFile]] = File(None),
    mask: Optional[UploadFile] = File(None),
):
    """Image-to-image generation.

    Accepts both the official ``multipart/form-data`` upload and a JSON body
    carrying data URLs or http(s) URLs.
    """
    try:
        content_type = (raw_request.headers.get("content-type") or "").lower()
        is_multipart = "multipart/form-data" in content_type

        passthrough: Dict[str, Any] = {}
        images: List[bytes] = []

        if is_multipart:
            resolved_prompt = (prompt or "").strip()
            images = await read_upload_files(image)
            requested_model = model
            requested_n = n
            requested_size = size
            requested_quality = quality
            requested_response_format = response_format
            requested_output_format = output_format
            requested_background = background
            if mask is not None:
                debug_logger.log_warning("[IMAGES] 上游不支持 mask，已忽略该字段")
        else:
            passthrough = await _read_json_body(raw_request)
            payload = ImageEditRequest.model_validate(passthrough)
            resolved_prompt = (payload.prompt or "").strip()
            images = await load_image_bytes(payload.image)
            requested_model = payload.model
            requested_n = payload.n
            requested_size = payload.size
            requested_quality = payload.quality
            requested_response_format = payload.response_format
            requested_output_format = payload.output_format
            requested_background = payload.background
            if payload.mask is not None:
                debug_logger.log_warning("[IMAGES] 上游不支持 mask，已忽略该字段")

        if not resolved_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if not images:
            raise HTTPException(status_code=400, detail="At least one image is required")

        images = images[:MAX_IMAGES]

        resolved_model = resolve_media_model(
            requested_model=requested_model,
            default_model=DEFAULT_IMAGE_MODEL,
            size=requested_size,
            quality=requested_quality,
            images=images,
            passthrough=passthrough,
        )
        assert_model_type(resolved_model, "image")

        count = min(max(requested_n or 1, 1), MAX_IMAGES)
        params = MediaParams(
            model=resolved_model,
            prompt=resolved_prompt,
            images=images,
            size=requested_size,
            quality=requested_quality,
        )

        data = await _generate_images(
            params,
            count=count,
            response_format=requested_response_format or "url",
            base_url=get_request_base_url(raw_request),
        )
        return JSONResponse(
            content=_build_response(
                data,
                model=resolved_model,
                size=requested_size,
                quality=requested_quality,
                output_format=requested_output_format,
                background=requested_background,
            )
        )

    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.status_code, str(exc.detail)),
        )
    except Exception as exc:
        debug_logger.log_error(f"[IMAGES] 编辑异常: {exc}")
        return JSONResponse(status_code=500, content=build_error_payload(500, str(exc)))
