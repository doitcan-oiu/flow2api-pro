"""Native OpenAI videos protocol.

Endpoints:

- ``POST   /v1/videos``                    create a video job
- ``GET    /v1/videos``                    list jobs
- ``GET    /v1/videos/{video_id}``         poll job status
- ``GET    /v1/videos/{video_id}/content`` download the rendered video
- ``POST   /v1/videos/{video_id}/remix``   re-render with a new prompt
- ``DELETE /v1/videos/{video_id}``         drop a job

Video generation upstream takes minutes, so this follows the official async
job model: create returns immediately with ``status: queued`` and the client
polls until ``completed``.
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ..core.auth import verify_api_key_flexible
from ..core.logger import debug_logger
from ..core.media_models import (
    VIDEO_STATUS_COMPLETED,
    VideoCreateRequest,
    VideoRemixRequest,
    build_video_list_payload,
)
from ..services.video_jobs import VideoJob, video_job_store
from .media_common import (
    DEFAULT_VIDEO_MODEL,
    MediaParams,
    assert_model_type,
    build_error_payload,
    get_request_base_url,
    load_image_bytes,
    normalize_seconds,
    read_media_bytes,
    read_upload_files,
    resolve_media_model,
    run_generation,
)

router = APIRouter(prefix="/v1/videos", tags=["videos"])


def _describe_image_range(min_images: int, max_images: Optional[int]) -> str:
    """Render an image-count requirement as `1`, `1-2` or `up to 3`."""
    if max_images is None:
        return f"at least {min_images}"
    if min_images == max_images:
        return str(max_images)
    if min_images <= 0:
        return f"up to {max_images}"
    return f"{min_images}-{max_images}"


async def _read_json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


async def _run_job(job: VideoJob, params: MediaParams, base_url: Optional[str]) -> None:
    """Background worker that drives one video job to completion."""
    video_job_store.mark_running(job)
    try:
        url = await run_generation(params, base_url_override=base_url)
        video_job_store.mark_completed(job, url)
        debug_logger.log_info(f"[VIDEO_JOB] {job.id} 完成: {url}")
    except asyncio.CancelledError:
        video_job_store.mark_failed(job, "Video job cancelled", code="cancelled")
        raise
    except HTTPException as exc:
        video_job_store.mark_failed(job, str(exc.detail))
    except Exception as exc:
        video_job_store.mark_failed(job, str(exc))


async def _start_job(
    *,
    prompt: str,
    requested_model: Optional[str],
    seconds: Optional[str],
    size: Optional[str],
    images: List[bytes],
    passthrough: Dict[str, Any],
    base_url: Optional[str],
    quality: Optional[str] = None,
    remixed_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate input, register the job and kick off background generation."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    normalized_seconds = normalize_seconds(seconds)

    model = resolve_media_model(
        requested_model=requested_model,
        default_model=DEFAULT_VIDEO_MODEL,
        size=size,
        quality=quality,
        seconds=normalized_seconds,
        images=images,
        passthrough=passthrough,
    )
    model_config = assert_model_type(model, "video")

    # Reference images only make sense for i2v/interpolation style models.
    if images and not model_config.get("supports_images"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' does not accept an input reference; "
                "use an image-to-video model instead"
            ),
        )

    # Reject counts the model cannot honour instead of silently dropping
    # images: quietly truncating changes what the user asked for, and they
    # would only notice by inspecting the rendered video.
    min_images = int(model_config.get("min_images") or 0)
    max_images = model_config.get("max_images")
    if not isinstance(max_images, int) or max_images <= 0:
        max_images = None

    if min_images and len(images) < min_images:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' requires {_describe_image_range(min_images, max_images)} "
                f"input reference image(s), got {len(images)}"
            ),
        )

    if max_images is not None and len(images) > max_images:
        hint = ""
        if max_images == 1:
            # The common case: a first-frame model given both a first frame
            # and a character/style reference.
            hint = (
                ". This model only accepts a first frame; use an interpolation "
                "model for first+last frame, or a multi-reference (r2v) model "
                "for several reference images"
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' accepts {_describe_image_range(min_images, max_images)} "
                f"input reference image(s), got {len(images)}{hint}"
            ),
        )

    job = await video_job_store.create(
        model=model,
        prompt=prompt,
        seconds=normalized_seconds,
        size=size,
        quality=quality,
        remixed_from_video_id=remixed_from,
    )
    params = MediaParams(
        model=model,
        prompt=prompt,
        images=images,
        size=size,
        seconds=normalized_seconds,
        quality=quality,
    )
    job.task = asyncio.create_task(_run_job(job, params, base_url))
    debug_logger.log_info(f"[VIDEO_JOB] {job.id} 已创建 - 模型: {model}")
    return job.to_resource(base_url=base_url)


@router.post("")
@router.post("/")
async def create_video(
    raw_request: Request,
    api_key: str = Depends(verify_api_key_flexible),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    input_reference: Optional[List[UploadFile]] = File(None),
):
    """Create a video generation job."""
    try:
        content_type = (raw_request.headers.get("content-type") or "").lower()
        passthrough: Dict[str, Any] = {}

        if "multipart/form-data" in content_type:
            resolved_prompt = prompt or ""
            images = await read_upload_files(input_reference)
            requested_model = model
            requested_seconds = seconds
            requested_size = size
            requested_quality = quality
        else:
            passthrough = await _read_json_body(raw_request)
            payload = VideoCreateRequest.model_validate(passthrough)
            resolved_prompt = payload.prompt
            images = await load_image_bytes(payload.input_reference)
            requested_model = payload.model
            requested_seconds = payload.seconds
            requested_size = payload.size
            requested_quality = payload.quality

        resource = await _start_job(
            prompt=resolved_prompt,
            requested_model=requested_model,
            seconds=requested_seconds,
            size=requested_size,
            quality=requested_quality,
            images=images,
            passthrough=passthrough,
            base_url=get_request_base_url(raw_request),
        )
        return JSONResponse(content=resource)

    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.status_code, str(exc.detail)),
        )
    except Exception as exc:
        debug_logger.log_error(f"[VIDEOS] 创建异常: {exc}")
        return JSONResponse(status_code=500, content=build_error_payload(500, str(exc)))


@router.get("")
@router.get("/")
async def list_videos(
    raw_request: Request,
    limit: int = Query(20, ge=1, le=100),
    order: str = Query("desc"),
    api_key: str = Depends(verify_api_key_flexible),
):
    """List known video jobs, newest first by default."""
    jobs = await video_job_store.list(limit=limit, order=order)
    base_url = get_request_base_url(raw_request)
    return JSONResponse(
        content=build_video_list_payload(
            [job.to_resource(base_url=base_url) for job in jobs]
        )
    )


async def _require_job(video_id: str) -> VideoJob:
    job = await video_job_store.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    return job


@router.get("/{video_id}")
async def retrieve_video(
    video_id: str,
    raw_request: Request,
    api_key: str = Depends(verify_api_key_flexible),
):
    """Poll a video job."""
    try:
        job = await _require_job(video_id)
        return JSONResponse(
            content=job.to_resource(base_url=get_request_base_url(raw_request))
        )
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.status_code, str(exc.detail)),
        )


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str,
    api_key: str = Depends(verify_api_key_flexible),
):
    """Download the rendered video bytes."""
    try:
        job = await _require_job(video_id)

        if job.status != VIDEO_STATUS_COMPLETED or not job.url:
            raise HTTPException(
                status_code=409,
                detail=f"Video is not ready (status: {job.status})",
            )

        content = await read_media_bytes(job.url)
        if not content:
            raise HTTPException(status_code=502, detail="Failed to read generated video")

        return Response(
            content=content,
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="{job.id}.mp4"'},
        )
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.status_code, str(exc.detail)),
        )
    except Exception as exc:
        debug_logger.log_error(f"[VIDEOS] 下载异常: {exc}")
        return JSONResponse(status_code=500, content=build_error_payload(500, str(exc)))


@router.post("/{video_id}/remix")
async def remix_video(
    video_id: str,
    request: VideoRemixRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key_flexible),
):
    """Re-render a completed video with an updated prompt."""
    try:
        source = await _require_job(video_id)
        if source.status != VIDEO_STATUS_COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Only completed videos can be remixed (status: {source.status})",
            )

        passthrough = await _read_json_body(raw_request)
        resource = await _start_job(
            prompt=request.prompt,
            requested_model=request.model or source.model,
            seconds=request.seconds or source.seconds,
            size=request.size or source.size,
            quality=request.quality or source.quality,
            images=[],
            passthrough=passthrough,
            base_url=get_request_base_url(raw_request),
            remixed_from=source.id,
        )
        return JSONResponse(content=resource)

    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.status_code, str(exc.detail)),
        )
    except Exception as exc:
        debug_logger.log_error(f"[VIDEOS] Remix 异常: {exc}")
        return JSONResponse(status_code=500, content=build_error_payload(500, str(exc)))


@router.delete("/{video_id}")
async def delete_video(video_id: str, api_key: str = Depends(verify_api_key_flexible)):
    """Delete a video job and cancel it if still running."""
    job = await video_job_store.delete(video_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=build_error_payload(404, f"Video not found: {video_id}"),
        )
    return JSONResponse(content={"id": video_id, "object": "video.deleted", "deleted": True})
