"""Tests for the native media protocols: /v1/images and /v1/videos."""

import asyncio
import base64
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import images as images_api
from src.api import media_common
from src.api import videos as videos_api
from src.core.config import config
from src.services.generation_handler import MODEL_CONFIG
from src.services.video_jobs import VideoJobStore

API_KEY = "test-media-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


class FakeHandler:
    """Minimal GenerationHandler stand-in recording calls and returning URLs."""

    def __init__(self, url: str = "http://testserver/tmp/out.png", error: Optional[Dict] = None):
        self.url = url
        self.error = error
        self.calls: List[Dict[str, Any]] = []
        self.file_cache = None

    async def handle_generation(
        self,
        model: str,
        prompt: str,
        images=None,
        stream: bool = False,
        base_url_override: Optional[str] = None,
        video_media_id: Optional[str] = None,
    ):
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "images": list(images or []),
                "stream": stream,
                "base_url_override": base_url_override,
            }
        )
        await asyncio.sleep(0)
        if self.error is not None:
            import json

            yield json.dumps({"error": self.error})
            return

        media_type = MODEL_CONFIG.get(model, {}).get("type")
        if media_type == "video":
            content = f"```html\n<video src='{self.url}' controls></video>\n```"
        else:
            content = f"![Generated Image]({self.url})"

        import json

        yield json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}}
                ],
            }
        )


def build_client(handler: FakeHandler) -> TestClient:
    app = FastAPI()
    app.include_router(images_api.router)
    app.include_router(videos_api.router)
    media_common.set_generation_handler(handler)
    return TestClient(app)


class MediaEndpointTestCase(unittest.TestCase):
    def setUp(self):
        self._original_api_key = config.api_key
        config.api_key = API_KEY
        # Isolate job state per test.
        self._original_store = videos_api.video_job_store
        videos_api.video_job_store = VideoJobStore()

    def tearDown(self):
        config.api_key = self._original_api_key
        videos_api.video_job_store = self._original_store
        media_common.set_generation_handler(None)


class ImageEndpointTests(MediaEndpointTestCase):
    def test_generations_returns_openai_envelope(self):
        handler = FakeHandler(url="http://testserver/tmp/img.png")
        client = build_client(handler)

        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={"model": "gemini-3.1-flash-image", "prompt": "a red fox"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("created", body)
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(body["data"][0]["url"], "http://testserver/tmp/img.png")
        self.assertEqual(handler.calls[0]["prompt"], "a red fox")
        self.assertFalse(handler.calls[0]["stream"])

    def test_generations_requires_auth(self):
        client = build_client(FakeHandler())
        response = client.post(
            "/v1/images/generations", json={"prompt": "hi", "model": "gemini-3.1-flash-image"}
        )
        self.assertEqual(response.status_code, 401)

    def test_generations_rejects_empty_prompt(self):
        client = build_client(FakeHandler())
        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={"prompt": "   ", "model": "gemini-3.1-flash-image"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_size_maps_to_portrait_model(self):
        handler = FakeHandler()
        client = build_client(handler)

        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={
                "model": "gemini-3.1-flash-image",
                "prompt": "portrait shot",
                "size": "1024x1792",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(handler.calls[0]["model"], "gemini-3.1-flash-image-portrait")

    def test_quality_high_maps_to_4k_variant(self):
        handler = FakeHandler()
        client = build_client(handler)

        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={
                "model": "gemini-3.1-flash-image",
                "prompt": "sharp",
                "size": "1024x1024",
                "quality": "high",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(handler.calls[0]["model"], "gemini-3.1-flash-image-square-4k")

    def test_n_generates_multiple_images(self):
        handler = FakeHandler()
        client = build_client(handler)

        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={"model": "gemini-3.1-flash-image", "prompt": "many", "n": 3},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 3)
        self.assertEqual(len(handler.calls), 3)

    def test_b64_json_response_format(self):
        handler = FakeHandler()
        client = build_client(handler)

        with patch.object(
            media_common, "read_media_bytes", return_value=PNG_BYTES
        ) as mocked:
            async def fake_read(url):
                return PNG_BYTES

            mocked.side_effect = fake_read
            with patch.object(images_api, "read_media_bytes", new=fake_read):
                response = client.post(
                    "/v1/images/generations",
                    headers=AUTH,
                    json={
                        "model": "gemini-3.1-flash-image",
                        "prompt": "inline",
                        "response_format": "b64_json",
                    },
                )

        self.assertEqual(response.status_code, 200)
        entry = response.json()["data"][0]
        self.assertIsNone(entry["url"])
        self.assertEqual(base64.b64decode(entry["b64_json"]), PNG_BYTES)

    def test_upstream_error_propagates_status(self):
        handler = FakeHandler(error={"message": "no token available", "status_code": 503})
        client = build_client(handler)

        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={"model": "gemini-3.1-flash-image", "prompt": "x"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["message"], "no token available")

    def test_video_model_rejected_on_image_endpoint(self):
        client = build_client(FakeHandler())
        response = client.post(
            "/v1/images/generations",
            headers=AUTH,
            json={"model": "veo_3_1_t2v_fast", "prompt": "x"},
        )
        self.assertEqual(response.status_code, 400)

    def test_edits_json_body_with_data_url(self):
        handler = FakeHandler()
        client = build_client(handler)

        response = client.post(
            "/v1/images/edits",
            headers=AUTH,
            json={
                "model": "gemini-3.1-flash-image",
                "prompt": "make it blue",
                "image": PNG_DATA_URL,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(handler.calls[0]["images"], [PNG_BYTES])

    def test_edits_multipart_upload(self):
        handler = FakeHandler()
        client = build_client(handler)

        response = client.post(
            "/v1/images/edits",
            headers=AUTH,
            data={"prompt": "edit me", "model": "gemini-3.1-flash-image"},
            files={"image": ("in.png", PNG_BYTES, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(handler.calls[0]["images"], [PNG_BYTES])

    def test_edits_requires_image(self):
        client = build_client(FakeHandler())
        response = client.post(
            "/v1/images/edits",
            headers=AUTH,
            json={"prompt": "no image", "model": "gemini-3.1-flash-image"},
        )
        self.assertEqual(response.status_code, 400)


class VideoEndpointTests(MediaEndpointTestCase):
    def _wait_for_status(self, client: TestClient, video_id: str, target: str, tries: int = 50) -> Dict:
        for _ in range(tries):
            body = client.get(f"/v1/videos/{video_id}", headers=AUTH).json()
            if body["status"] == target:
                return body
        raise AssertionError(f"job did not reach {target}, last={body}")

    def test_create_returns_queued_job(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        response = client.post(
            "/v1/videos",
            headers=AUTH,
            json={"model": "veo_3_1_t2v_fast", "prompt": "a cat surfing"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["id"].startswith("video_"))
        self.assertEqual(body["object"], "video")
        self.assertIn(body["status"], {"queued", "in_progress", "completed"})
        self.assertIsNone(body["error"])

    def test_job_reaches_completed_and_serves_content(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        created = client.post(
            "/v1/videos",
            headers=AUTH,
            json={"model": "veo_3_1_t2v_fast", "prompt": "a cat surfing"},
        ).json()

        done = self._wait_for_status(client, created["id"], "completed")
        self.assertEqual(done["progress"], 100)
        self.assertIsNotNone(done["completed_at"])

        async def fake_read(url):
            return b"MP4DATA"

        with patch.object(videos_api, "read_media_bytes", new=fake_read):
            content = client.get(f"/v1/videos/{created['id']}/content", headers=AUTH)

        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, b"MP4DATA")
        self.assertEqual(content.headers["content-type"], "video/mp4")

    def test_content_conflict_before_completion(self):
        handler = FakeHandler(error={"message": "boom", "status_code": 500})
        client = build_client(handler)

        created = client.post(
            "/v1/videos", headers=AUTH, json={"model": "veo_3_1_t2v_fast", "prompt": "x"}
        ).json()
        self._wait_for_status(client, created["id"], "failed")

        response = client.get(f"/v1/videos/{created['id']}/content", headers=AUTH)
        self.assertEqual(response.status_code, 409)

    def test_failed_job_exposes_error(self):
        handler = FakeHandler(error={"message": "upstream down", "status_code": 502})
        client = build_client(handler)

        created = client.post(
            "/v1/videos", headers=AUTH, json={"model": "veo_3_1_t2v_fast", "prompt": "x"}
        ).json()
        failed = self._wait_for_status(client, created["id"], "failed")

        self.assertEqual(failed["error"]["message"], "upstream down")

    def test_seconds_selects_duration_variant(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        created = client.post(
            "/v1/videos",
            headers=AUTH,
            json={"model": "veo_3_1_t2v_fast", "prompt": "x", "seconds": "8"},
        ).json()
        self._wait_for_status(client, created["id"], "completed")

        self.assertEqual(handler.calls[0]["model"], "veo_3_1_t2v_fast_8s")
        self.assertEqual(created["seconds"], "8")

    def test_size_portrait_selects_portrait_model(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        created = client.post(
            "/v1/videos",
            headers=AUTH,
            json={"model": "veo_3_1_t2v_fast", "prompt": "x", "size": "720x1280"},
        ).json()
        self._wait_for_status(client, created["id"], "completed")

        self.assertEqual(handler.calls[0]["model"], "veo_3_1_t2v_fast_portrait")

    def test_input_reference_rejected_for_t2v_model(self):
        client = build_client(FakeHandler())
        response = client.post(
            "/v1/videos",
            headers=AUTH,
            json={
                "model": "veo_3_1_t2v_fast",
                "prompt": "x",
                "input_reference": PNG_DATA_URL,
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_input_reference_accepted_for_i2v_model(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        created = client.post(
            "/v1/videos",
            headers=AUTH,
            json={
                "model": "veo_3_1_i2v_s_fast_fl",
                "prompt": "animate this",
                "input_reference": PNG_DATA_URL,
            },
        ).json()
        self._wait_for_status(client, created["id"], "completed")

        self.assertEqual(handler.calls[0]["images"], [PNG_BYTES])

    def test_image_model_rejected_on_video_endpoint(self):
        client = build_client(FakeHandler())
        response = client.post(
            "/v1/videos",
            headers=AUTH,
            json={"model": "gemini-3.1-flash-image", "prompt": "x"},
        )
        self.assertEqual(response.status_code, 400)

    def test_retrieve_unknown_video_returns_404(self):
        client = build_client(FakeHandler())
        response = client.get("/v1/videos/video_missing", headers=AUTH)
        self.assertEqual(response.status_code, 404)

    def test_list_and_delete(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        created = client.post(
            "/v1/videos", headers=AUTH, json={"model": "veo_3_1_t2v_fast", "prompt": "x"}
        ).json()
        self._wait_for_status(client, created["id"], "completed")

        listing = client.get("/v1/videos", headers=AUTH).json()
        self.assertEqual(listing["object"], "list")
        self.assertEqual(len(listing["data"]), 1)

        deleted = client.delete(f"/v1/videos/{created['id']}", headers=AUTH)
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])

        self.assertEqual(
            client.get(f"/v1/videos/{created['id']}", headers=AUTH).status_code, 404
        )

    def test_remix_requires_completed_source(self):
        handler = FakeHandler(error={"message": "nope", "status_code": 500})
        client = build_client(handler)

        created = client.post(
            "/v1/videos", headers=AUTH, json={"model": "veo_3_1_t2v_fast", "prompt": "x"}
        ).json()
        self._wait_for_status(client, created["id"], "failed")

        response = client.post(
            f"/v1/videos/{created['id']}/remix", headers=AUTH, json={"prompt": "new"}
        )
        self.assertEqual(response.status_code, 409)

    def test_remix_creates_linked_job(self):
        handler = FakeHandler(url="http://testserver/tmp/out.mp4")
        client = build_client(handler)

        created = client.post(
            "/v1/videos", headers=AUTH, json={"model": "veo_3_1_t2v_fast", "prompt": "x"}
        ).json()
        self._wait_for_status(client, created["id"], "completed")

        remix = client.post(
            f"/v1/videos/{created['id']}/remix", headers=AUTH, json={"prompt": "brighter"}
        )
        self.assertEqual(remix.status_code, 200)
        body = remix.json()
        self.assertEqual(body["remixed_from_video_id"], created["id"])
        self.assertNotEqual(body["id"], created["id"])

    def test_videos_requires_auth(self):
        client = build_client(FakeHandler())
        response = client.post("/v1/videos", json={"prompt": "x", "model": "veo_3_1_t2v_fast"})
        self.assertEqual(response.status_code, 401)


class SecondsNormalizationTests(unittest.TestCase):
    def test_exact_values_pass_through(self):
        self.assertEqual(media_common.normalize_seconds("8"), "8")
        self.assertEqual(media_common.normalize_seconds("4s"), "4")

    def test_snaps_to_nearest_supported(self):
        self.assertEqual(media_common.normalize_seconds("5"), "4")
        self.assertEqual(media_common.normalize_seconds("12"), "8")

    def test_invalid_values_return_none(self):
        self.assertIsNone(media_common.normalize_seconds(None))
        self.assertIsNone(media_common.normalize_seconds(""))
        self.assertIsNone(media_common.normalize_seconds("abc"))
        self.assertIsNone(media_common.normalize_seconds("0"))


if __name__ == "__main__":
    unittest.main()
