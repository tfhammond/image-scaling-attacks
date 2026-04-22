import base64
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
import cv2

backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

import app as backend_app


def encode_image(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_decoy(width: int, height: int) -> str:
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            gradient[y, x] = [(x * 13) % 255, (y * 17) % 255, ((x + y) * 11) % 255]
    return encode_image(gradient)


def decode_data_url_image(data_url: str) -> Image.Image:
    encoded = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def load_reference_downsampler_class(module_name: str, class_name: str):
    reference_dir = (
        Path(__file__).resolve().parent.parent / "anamorpher" / "backend" / "downsamplers"
    )
    package_name = "_anamorpher_reference_downsamplers"

    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(reference_dir)]
        sys.modules[package_name] = package

    for dependency_module in ("base", module_name):
        qualified_name = f"{package_name}.{dependency_module}"
        if qualified_name in sys.modules:
            continue

        module_path = reference_dir / f"{dependency_module}.py"
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)

    return getattr(sys.modules[f"{package_name}.{module_name}"], class_name)


@pytest.fixture()
def client():
    backend_app.app.config.update(TESTING=True)
    return backend_app.app.test_client()


class TestBackendApp:
    def test_local_demo_downsamplers_match_original_references(self):
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)

        original_tensorflow = load_reference_downsampler_class(
            "tensorflow_downsampler", "TensorFlowDownsampler"
        )()
        nearest_expected = original_tensorflow.downsample(image, (4, 4), "nearest")
        nearest_actual = backend_app.downsample_adversarial_image(image, "nearest")
        assert np.array_equal(nearest_actual, nearest_expected)

        original_opencv = load_reference_downsampler_class(
            "opencv_downsampler", "OpenCVDownsampler"
        )()
        bicubic_expected = original_opencv.downsample(image, (4, 4), "bicubic")
        bicubic_actual = backend_app.downsample_adversarial_image(image, "bicubic")
        assert np.array_equal(bicubic_actual, bicubic_expected)

        bilinear_expected = cv2.resize(image, (4, 4), interpolation=cv2.INTER_LINEAR_EXACT)
        bilinear_actual = backend_app.downsample_adversarial_image(image, "bilinear", {"anti_alias": False})
        assert np.array_equal(bilinear_actual, bilinear_expected)

        bilinear_aa_expected = cv2.resize(image, (4, 4), interpolation=cv2.INTER_LINEAR)
        bilinear_aa_actual = backend_app.downsample_adversarial_image(image, "bilinear", {"anti_alias": True})
        assert np.array_equal(bilinear_aa_actual, bilinear_aa_expected)

    def test_index_serves_frontend(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert b"Image Scaling Prompt Injection Attacks Project" in response.data
        assert b"Generate" in response.data

    def test_frontend_assets_are_served(self, client):
        script_response = client.get("/frontend/script.js")
        style_response = client.get("/frontend/styles.css")

        assert script_response.status_code == 200
        assert b"METHOD_DEFAULTS" in script_response.data
        assert b"bilinear" in script_response.data
        assert style_response.status_code == 200
        assert b".result-grid" in style_response.data

    def test_health(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.get_json() == {"status": "healthy"}

    def test_preview_text_returns_image_and_size(self, client):
        response = client.post(
            "/preview-text",
            json={"text": "hello world", "font_size": 32, "alignment": "center"},
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["image"].startswith("data:image/png;base64,")
        assert isinstance(payload["text_overflowed"], bool)
        assert payload["size"] == "1092x1092"

    def test_sanitize_text_preserves_at_character(self):
        assert backend_app.sanitize_text("hello@example.com") == "hello@example.com"

    def test_wrap_text_to_fit_breaks_long_text_to_width(self):
        image = Image.new("RGB", (256, 256), color="#333333")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        max_width = 80

        lines = backend_app.wrap_text_to_fit(
            "Supercalifragilisticexpialidocious wrapped across the image width",
            font,
            draw,
            max_width,
        )

        assert len(lines) > 1
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            assert bbox[2] - bbox[0] <= max_width

    def test_wrap_text_to_fit_preserves_explicit_line_breaks(self):
        image = Image.new("RGB", (256, 256), color="#333333")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        lines = backend_app.wrap_text_to_fit("alpha\nbeta gamma", font, draw, 300)

        assert lines == ["alpha", "beta gamma"]

    def test_preview_text_respects_explicit_size(self, client):
        response = client.post(
            "/preview-text",
            json={"text": "hello", "font_size": 32, "alignment": "center", "size": 256},
        )

        assert response.status_code == 200
        assert response.get_json()["size"] == "256x256"

    def test_preview_text_rejects_invalid_alignment(self, client):
        response = client.post(
            "/preview-text",
            json={"text": "hello", "font_size": 32, "alignment": "diagonal"},
        )

        assert response.status_code == 400
        assert "Invalid alignment" in response.get_json()["error"]

    def test_preview_text_rejects_invalid_numeric_input(self, client):
        response = client.post(
            "/preview-text",
            json={"text": "hello", "font_size": "big", "alignment": "center"},
        )

        assert response.status_code == 400
        assert "Invalid numeric value" in response.get_json()["error"]

    def test_preview_text_rejects_too_long_text(self, client):
        response = client.post(
            "/preview-text",
            json={"text": "a" * 1001, "font_size": 32, "alignment": "center"},
        )

        assert response.status_code == 400
        assert "Text too long" in response.get_json()["error"]

    def test_generate_adversarial_nearest(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "nearest",
                "font_size": 32,
                "alignment": "center",
                "lam": 0.25,
                "eps": 0.0,
                "gamma": 1.0,
                "offset": 2,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["method"] == "nearest"
        assert payload["parameters"]["offset"] == 2
        assert "dark_frac" not in payload["parameters"]
        assert payload["adversarial_image"].startswith("data:image/png;base64,")
        assert payload["downsampled_image"].startswith("data:image/png;base64,")
        assert payload["target_image"].startswith("data:image/png;base64,")
        assert payload["adversarial_size"] == "16x16"
        assert payload["downsampled_size"] == "4x4"
        assert payload["target_size"] == "4x4"

        adversarial = decode_data_url_image(payload["adversarial_image"])
        downsampled = decode_data_url_image(payload["downsampled_image"])
        original_downsampler = load_reference_downsampler_class(
            "tensorflow_downsampler", "TensorFlowDownsampler"
        )()
        expected = original_downsampler.downsample(
            np.array(adversarial), downsampled.size, "nearest"
        )
        assert np.array_equal(np.array(downsampled), expected)

    def test_generate_adversarial_bicubic(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "bicubic",
                "font_size": 32,
                "alignment": "center",
                "lam": 0.25,
                "eps": 0.0,
                "gamma": 1.0,
                "dark_frac": 0.3,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["method"] == "bicubic"
        assert payload["parameters"]["dark_frac"] == pytest.approx(0.3)
        assert "offset" not in payload["parameters"]
        assert payload["adversarial_image"].startswith("data:image/png;base64,")
        assert payload["downsampled_image"].startswith("data:image/png;base64,")
        assert payload["target_image"].startswith("data:image/png;base64,")
        assert payload["adversarial_size"] == "16x16"
        assert payload["downsampled_size"] == "4x4"
        assert payload["target_size"] == "4x4"

        adversarial = decode_data_url_image(payload["adversarial_image"])
        downsampled = decode_data_url_image(payload["downsampled_image"])
        original_downsampler = load_reference_downsampler_class(
            "opencv_downsampler", "OpenCVDownsampler"
        )()
        expected = original_downsampler.downsample(
            np.array(adversarial), downsampled.size, "bicubic"
        )
        assert np.array_equal(np.array(downsampled), expected)

    def test_generate_adversarial_bilinear(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "bilinear",
                "font_size": 32,
                "alignment": "center",
                "lam": 1.0,
                "eps": 0.0,
                "gamma": 0.9,
                "dark_frac": 0.3,
                "anti_alias": False,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["method"] == "bilinear"
        assert payload["parameters"]["dark_frac"] == pytest.approx(0.3)
        assert payload["parameters"]["anti_alias"] is False
        assert "offset" not in payload["parameters"]
        assert payload["adversarial_image"].startswith("data:image/png;base64,")
        assert payload["downsampled_image"].startswith("data:image/png;base64,")
        assert payload["target_image"].startswith("data:image/png;base64,")

        adversarial = decode_data_url_image(payload["adversarial_image"])
        downsampled = decode_data_url_image(payload["downsampled_image"])
        expected = cv2.resize(
            np.array(adversarial),
            downsampled.size,
            interpolation=cv2.INTER_LINEAR_EXACT,
        )
        assert np.array_equal(np.array(downsampled), expected)

    def test_generate_adversarial_bilinear_with_antialias(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "bilinear",
                "font_size": 32,
                "alignment": "center",
                "lam": 1.0,
                "eps": 0.0,
                "gamma": 0.9,
                "dark_frac": 0.3,
                "anti_alias": True,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["parameters"]["anti_alias"] is True

        adversarial = decode_data_url_image(payload["adversarial_image"])
        downsampled = decode_data_url_image(payload["downsampled_image"])
        expected = cv2.resize(
            np.array(adversarial),
            downsampled.size,
            interpolation=cv2.INTER_LINEAR,
        )
        assert np.array_equal(np.array(downsampled), expected)

    def test_generate_adversarial_rejects_missing_image(self, client):
        response = client.post(
            "/generate-adversarial",
            json={"text": "hide", "method": "nearest", "font_size": 32, "alignment": "center"},
        )

        assert response.status_code == 400
        assert response.get_json()["error"] == "Decoy image is required"

    def test_generate_adversarial_rejects_non_square_decoy(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(12, 16),
                "text": "hide",
                "method": "nearest",
                "font_size": 32,
                "alignment": "center",
            },
        )

        assert response.status_code == 400
        assert "must be square" in response.get_json()["error"]

    def test_generate_adversarial_rejects_decoy_not_divisible_by_four(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(18, 18),
                "text": "hide",
                "method": "nearest",
                "font_size": 32,
                "alignment": "center",
            },
        )

        assert response.status_code == 400
        assert "divisible by 4" in response.get_json()["error"]

    def test_generate_adversarial_rejects_unsupported_method(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "lanczos",
                "font_size": 32,
                "alignment": "center",
            },
        )

        assert response.status_code == 400
        assert "Invalid method" in response.get_json()["error"]

    def test_generate_adversarial_rejects_invalid_base64(self, client):
        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": "not-base64",
                "text": "hide",
                "method": "nearest",
                "font_size": 32,
                "alignment": "center",
            },
        )

        assert response.status_code == 400
        assert "Invalid base64 format" in response.get_json()["error"]

    def test_generate_adversarial_handles_subprocess_failure(self, client, monkeypatch):
        def fail_run(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        monkeypatch.setattr(backend_app.subprocess, "run", fail_run)

        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "nearest",
                "font_size": 32,
                "alignment": "center",
            },
        )

        assert response.status_code == 500
        assert response.get_json()["error"] == "Image generation failed (nearest method)"

    def test_generate_adversarial_handles_bilinear_subprocess_failure(self, client, monkeypatch):
        def fail_run(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        monkeypatch.setattr(backend_app.subprocess, "run", fail_run)

        response = client.post(
            "/generate-adversarial",
            json={
                "decoy_image": make_decoy(16, 16),
                "text": "hide",
                "method": "bilinear",
                "font_size": 32,
                "alignment": "center",
            },
        )

        assert response.status_code == 500
        assert response.get_json()["error"] == "Image generation failed (bilinear method)"
