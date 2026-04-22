import base64
import io
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from downsamplers import OpenCVDownsampler, TensorFlowDownsampler
from PIL import Image, ImageDraw, ImageFont
from sanitizer import (
    escape_for_html,
    sanitize_alignment,
    sanitize_method,
    sanitize_numeric,
    sanitize_text,
    validate_safe_path,
)
from werkzeug.exceptions import HTTPException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

BASE_DIR = Path(__file__).resolve().parent
GENERATORS_DIR = BASE_DIR / "adversarial_generators"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DEFAULT_PREVIEW_SIZE = 1092
MIN_IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 2048

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/frontend")


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Global handler for unhandled exceptions."""
    if isinstance(error, HTTPException):
        return error

    app.logger.exception("Unhandled exception in %s %s", request.method, request.path)
    return jsonify({"error": "An unexpected error occurred"}), 500


def resolve_script_path(env_key: str, default_name: str) -> str:
    raw_path = os.getenv(env_key)
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
    else:
        candidate = (GENERATORS_DIR / default_name).resolve()

    return validate_safe_path(str(candidate), str(GENERATORS_DIR))


BICUBIC_SCRIPT_PATH = resolve_script_path("BICUBIC_SCRIPT_PATH", "bicubic_gen_payload.py")
NEAREST_SCRIPT_PATH = resolve_script_path("NEAREST_SCRIPT_PATH", "nearest_gen_payload.py")
BILINEAR_SCRIPT_PATH = resolve_script_path("BILINEAR_SCRIPT_PATH", "bilinear_gen_payload.py")
PREVIEW_DOWNSAMPLERS = {
    "bicubic": OpenCVDownsampler(),
    "bilinear": OpenCVDownsampler(),
    "nearest": TensorFlowDownsampler(),
}


def image_to_base64(image: np.ndarray) -> str:
    """Convert a numpy image to a PNG data URL."""
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert a base64 string to a validated RGB image."""
    if not isinstance(base64_string, str):
        raise ValueError("Image data must be a string")
    if not base64_string:
        raise ValueError("Image data cannot be empty")
    if len(base64_string) > 15_000_000:
        raise ValueError("Image data too large")

    try:
        if base64_string.startswith("data:image"):
            if "," not in base64_string:
                raise ValueError("Invalid data URL format")
            base64_string = base64_string.split(",", 1)[1]

        if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", base64_string):
            raise ValueError("Invalid base64 format")

        image_bytes = base64.b64decode(base64_string, validate=True)
        if len(image_bytes) > 50_000_000:
            raise ValueError("Decoded image too large")

        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.format not in {"PNG", "JPEG", "JPG", "BMP", "TIFF"}:
            raise ValueError("Unsupported image format")
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        return np.array(pil_image)
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"Failed to decode image: {error}") from error


def create_text_image(
    text: str, size: int = DEFAULT_PREVIEW_SIZE, font_size: int = 32, alignment: str = "center"
) -> tuple[np.ndarray, bool]:
    """Create a square text image and report whether the text overflowed."""
    image = Image.new("RGB", (size, size), color="#333333")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except OSError:
        font_paths = [
            Path("/System/Library/Fonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        font = None
        for font_path in font_paths:
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), font_size)
                break
        if font is None:
            font = ImageFont.load_default()

    margin = 10
    text_area_width = size - 2 * margin
    text_area_height = size - 2 * margin
    wrapped_lines = wrap_text_to_fit(text, font, draw, text_area_width)

    line_height = (
        draw.textbbox((0, 0), "Ay", font=font)[3] - draw.textbbox((0, 0), "Ay", font=font)[1]
    )
    total_height = len(wrapped_lines) * line_height
    text_overflowed = total_height > text_area_height

    if alignment in ["center", "top", "bottom"]:
        if alignment == "center":
            start_y = max(margin, (size - total_height) // 2)
        elif alignment == "top":
            start_y = margin
        else:
            start_y = max(margin, size - margin - total_height)

        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break
            y = start_y + i * line_height
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (size - line_width) // 2
            draw.text((x, y), line, font=font, fill="#00b002")
    elif alignment in ["left", "right"]:
        start_y = max(margin, (size - total_height) // 2)
        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break
            y = start_y + i * line_height
            if alignment == "left":
                x = margin
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = size - margin - line_width
            draw.text((x, y), line, font=font, fill="#00b002")
    else:
        if alignment.startswith("top"):
            start_y = margin
        else:
            start_y = max(margin, size - margin - total_height)

        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break
            y = start_y + i * line_height
            if alignment.endswith("left"):
                x = margin
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = size - margin - line_width
            draw.text((x, y), line, font=font, fill="#00b002")

    return np.array(image), text_overflowed


def wrap_text_to_fit(text: str, font, draw: ImageDraw.ImageDraw, max_width: int) -> list[str]:
    """Wrap text to fit within the provided width."""
    def text_width(value: str) -> int:
        bbox = draw.textbbox((0, 0), value, font=font)
        return bbox[2] - bbox[0]

    def split_long_token(token: str) -> list[str]:
        if not token:
            return [""]

        parts: list[str] = []
        remaining = token
        while remaining:
            best_length = 0
            for index in range(1, len(remaining) + 1):
                if text_width(remaining[:index]) <= max_width:
                    best_length = index
                else:
                    break

            if best_length == 0:
                best_length = 1

            parts.append(remaining[:best_length])
            remaining = remaining[best_length:]

        return parts

    wrapped_lines: list[str] = []
    paragraphs = text.splitlines() or [text]

    for paragraph in paragraphs:
        if not paragraph.strip():
            wrapped_lines.append("")
            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            segments = [word]
            if text_width(word) > max_width:
                segments = split_long_token(word)

            for segment in segments:
                candidate = segment if not current_line else f"{current_line} {segment}"
                if text_width(candidate) <= max_width:
                    current_line = candidate
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = segment

        if current_line:
            wrapped_lines.append(current_line)

    return wrapped_lines or [""]


def parse_script_error(stderr: str, method: str) -> str:
    """Extract a user-friendly message from a generator script failure."""
    stderr_lower = stderr.lower()
    if "4x target size" in stderr_lower or "4× target" in stderr_lower:
        return "Decoy image must be exactly 4x the target resolution"
    if "failed to read" in stderr_lower or "cannot identify image" in stderr_lower:
        return "Failed to read image file"
    return f"Image generation failed ({method} method)"


def require_json_object() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def format_size(width: int, height: int) -> str:
    return f"{width}x{height}"


def get_generator_parameters(data: dict, method: str) -> dict:
    if method == "bilinear":
        common = {
            "lam": sanitize_numeric(data.get("lam", 1.0), min_val=0.0, max_val=10.0),
            "eps": sanitize_numeric(data.get("eps", 0.0), min_val=0.0, max_val=1.0),
            "gamma": sanitize_numeric(data.get("gamma", 0.9), min_val=0.1, max_val=3.0),
        }
    else:
        common = {
            "lam": sanitize_numeric(data.get("lam", 0.25), min_val=0.0, max_val=10.0),
            "eps": sanitize_numeric(data.get("eps", 0.0), min_val=0.0, max_val=1.0),
            "gamma": sanitize_numeric(data.get("gamma", 1.0), min_val=0.1, max_val=3.0),
        }
    if method == "nearest":
        common["offset"] = sanitize_numeric(data.get("offset", 2), min_val=0, max_val=3, data_type=int)
    else:
        common["dark_frac"] = sanitize_numeric(
            data.get("dark_frac", 0.3), min_val=0.0, max_val=1.0
        )
        if method == "bilinear":
            common["anti_alias"] = bool(data.get("anti_alias", False))
    return common


def build_generator_command(
    script_path: str, method: str, decoy_path: Path, target_path: Path, parameters: dict
) -> list[str]:
    command = [
        sys.executable,
        script_path,
        "--decoy",
        str(decoy_path),
        "--target",
        str(target_path),
        "--lam",
        str(parameters["lam"]),
        "--eps",
        str(parameters["eps"]),
        "--gamma",
        str(parameters["gamma"]),
    ]
    if method == "nearest":
        command.extend(["--offset", str(parameters["offset"])])
    else:
        command.extend(["--dark-frac", str(parameters["dark_frac"])])
        if method == "bilinear" and parameters.get("anti_alias"):
            command.append("--anti-alias")
    return command


def downsample_adversarial_image(image: np.ndarray, method: str, parameters: dict | None = None) -> np.ndarray:
    height, width = image.shape[:2]
    downsampled_size = (width // 4, height // 4)
    downsampler = PREVIEW_DOWNSAMPLERS.get(method)
    if downsampler is None:
        raise ValueError(f"Unsupported method: {method}")
    if method == "bilinear":
        anti_alias = bool((parameters or {}).get("anti_alias", False))
        return downsampler.downsample_bilinear(image, downsampled_size, anti_alias=anti_alias)
    return downsampler.downsample(image, downsampled_size, method)


def get_generated_image_path(temp_dir: Path, method: str) -> str:
    if method == "nearest":
        candidates = sorted(
            path for path in temp_dir.glob("advNN*.png") if not path.name.endswith("_down.png")
        )
    elif method == "bilinear":
        candidates = sorted(temp_dir.glob("adv_bilinear*.png"))
    else:
        candidates = sorted(temp_dir.glob("adv*.png"))

    if not candidates:
        raise RuntimeError("No adversarial image was generated")

    return validate_safe_path(str(candidates[0]), str(temp_dir))


@app.route("/", methods=["GET"])
def index():
    """Serve the frontend entry page."""
    return app.send_static_file("index.html")


@app.route("/preview-text", methods=["POST"])
def preview_text():
    """Generate a text preview image."""
    try:
        data = require_json_object()
        text = sanitize_text(data.get("text", "Sample Text"))
        size = sanitize_numeric(
            data.get("size", DEFAULT_PREVIEW_SIZE),
            min_val=MIN_IMAGE_SIZE,
            max_val=MAX_IMAGE_SIZE,
            data_type=int,
        )
        font_size = sanitize_numeric(data.get("font_size", 32), min_val=20, max_val=64, data_type=int)
        alignment = sanitize_alignment(data.get("alignment", "center"))

        preview_image, text_overflowed = create_text_image(text, size, font_size, alignment)

        return jsonify(
            {
                "image": image_to_base64(preview_image),
                "text_overflowed": text_overflowed,
                "size": format_size(size, size),
            }
        )
    except ValueError as error:
        return jsonify({"error": escape_for_html(str(error))}), 400
    except Exception:
        app.logger.exception("Error in preview_text")
        return jsonify({"error": "Text preview generation failed"}), 500


@app.route("/generate-adversarial", methods=["POST"])
def generate_adversarial():
    """Generate an adversarial image from an uploaded decoy image."""
    try:
        data = require_json_object()
        decoy_image_data = data.get("decoy_image")
        if not decoy_image_data:
            return jsonify({"error": "Decoy image is required"}), 400

        method = sanitize_method(data.get("method", "bicubic"))
        text = sanitize_text(data.get("text", "Sample Text"))
        font_size = sanitize_numeric(data.get("font_size", 32), min_val=20, max_val=64, data_type=int)
        alignment = sanitize_alignment(data.get("alignment", "center"))
        parameters = get_generator_parameters(data, method)

        decoy_image = base64_to_image(decoy_image_data)
        height, width = decoy_image.shape[:2]
        if height != width:
            return jsonify({"error": f"Decoy image must be square, got {width}x{height}"}), 400
        if width % 4 != 0:
            return jsonify({"error": f"Decoy image size must be divisible by 4, got {width}x{height}"}), 400

        target_size = width // 4
        target_image, _ = create_text_image(text, target_size, font_size, alignment)

        if method == "bicubic":
            script_path = BICUBIC_SCRIPT_PATH
        elif method == "bilinear":
            script_path = BILINEAR_SCRIPT_PATH
        else:
            script_path = NEAREST_SCRIPT_PATH

        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            decoy_path = temp_dir / "decoy.png"
            target_path = temp_dir / "target.png"
            Image.fromarray(decoy_image.astype(np.uint8)).save(decoy_path)
            Image.fromarray(target_image.astype(np.uint8)).save(target_path)

            result = subprocess.run(
                build_generator_command(script_path, method, decoy_path, target_path, parameters),
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                app.logger.error(
                    "Script failed (exit %d): %s\nstderr: %s",
                    result.returncode,
                    script_path,
                    result.stderr,
                )
                return jsonify({"error": parse_script_error(result.stderr, method)}), 500

            adversarial_path = get_generated_image_path(temp_dir, method)
            with Image.open(adversarial_path) as adv_image_file:
                adversarial_image = np.array(adv_image_file.convert("RGB"))

        downsampled_image = downsample_adversarial_image(adversarial_image, method, parameters)

        return jsonify(
            {
                "adversarial_image": image_to_base64(adversarial_image),
                "downsampled_image": image_to_base64(downsampled_image),
                "target_image": image_to_base64(target_image),
                "adversarial_size": format_size(width, height),
                "downsampled_size": format_size(target_size, target_size),
                "target_size": format_size(target_size, target_size),
                "method": method,
                "parameters": parameters,
            }
        )
    except ValueError as error:
        return jsonify({"error": escape_for_html(str(error))}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 500
    except Exception:
        app.logger.exception("Error in generate_adversarial")
        return jsonify({"error": "Adversarial image generation failed"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    print("Starting backend...")
    debug_enabled = os.getenv("ANAMORPHER_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug_enabled, use_reloader=False, host="127.0.0.1", port=5000)
