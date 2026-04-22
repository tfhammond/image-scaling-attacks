# Image Scaling Prompt Injection Attacks

This project demonstrates image-scaling attacks: a high-resolution decoy image is modified so that it looks mostly like the decoy at full size, but reveals a target text image after a 4:1 downsampling operation.

The app has a Flask backend and a plain HTML/CSS/JavaScript frontend served from the same origin. The workflow is:

1. Enter target text.
2. Preview the rendered target text image.
3. Upload a square decoy image whose dimensions are divisible by 4.
4. Choose an adversarial generation method.
5. Generate the adversarial image and its downsampled preview.

## Project Layout

```text
backend/
  app.py                         Flask app, API routes, image helpers, subprocess orchestration
  sanitizer.py                   Input validation and sanitization
  adversarial_generators/        Method-specific adversarial image generators
  downsamplers/                  Downsampling implementations used for returned previews
frontend/
  index.html                     Single-page UI
  script.js                      Browser-side validation and API calls
  styles.css                     UI styling
tests/
  test_backend_app.py            Route, integration, and downsampling behavior tests
  test_*_gen_payload.py          Generator unit tests
anamorpher/
  ...                            Upstream reference material, not edited by this project
```

## Setup

This project uses Python `>=3.11,<3.14`. The dependency set includes Flask, Pillow, NumPy, OpenCV, TensorFlow, Bleach, MarkupSafe, and Pytest.

The recommended environment manager is `uv`.

```powershell
uv sync
```

Run the app:

```powershell
uv run python backend/app.py
```

Open the frontend:

```text
http://127.0.0.1:5000
```

The backend binds to `127.0.0.1:5000` and runs without Flask's debug reloader by default. To enable debug mode:

```powershell
$env:ANAMORPHER_DEBUG="1"
uv run python backend/app.py
```

Run tests:

```powershell
python -m pytest -q
```

On this UNC-backed workspace, `uv run pytest` may fail when NumPy loads compiled extensions from `.venv`. If that happens, use the active Python command above.

## API

### `GET /`

Serves `frontend/index.html`.

Static frontend files are served from:

```text
/frontend/<asset>
```

### `GET /health`

Returns:

```json
{
  "status": "healthy"
}
```

### `POST /preview-text`

Generates a square target text image before an adversarial image is created.

Request:

```json
{
  "text": "Hidden text",
  "font_size": 32,
  "alignment": "center",
  "size": 1092
}
```

`size` is optional and defaults to `1092`.

Response:

```json
{
  "image": "data:image/png;base64,...",
  "text_overflowed": false,
  "size": "1092x1092"
}
```

The backend sanitizes text, validates numeric values, validates alignment, renders text with Pillow, wraps long lines to the image width, and reports whether the rendered line stack exceeds the available image height.

### `POST /generate-adversarial`

Generates the full adversarial image, the downsampled preview, and the target text image.

Common request fields:

```json
{
  "decoy_image": "data:image/png;base64,...",
  "text": "Hidden text",
  "method": "bicubic",
  "font_size": 32,
  "alignment": "center"
}
```

Supported methods:

```text
nearest
bicubic
bilinear
```

`nearest` parameters:

```json
{
  "lam": 0.25,
  "eps": 0.0,
  "gamma": 1.0,
  "offset": 2
}
```

`bicubic` parameters:

```json
{
  "lam": 0.25,
  "eps": 0.0,
  "gamma": 1.0,
  "dark_frac": 0.3
}
```

`bilinear` parameters:

```json
{
  "lam": 1.0,
  "eps": 0.0,
  "gamma": 0.9,
  "dark_frac": 0.3,
  "anti_alias": false
}
```

Response:

```json
{
  "adversarial_image": "data:image/png;base64,...",
  "downsampled_image": "data:image/png;base64,...",
  "target_image": "data:image/png;base64,...",
  "adversarial_size": "4368x4368",
  "downsampled_size": "1092x1092",
  "target_size": "1092x1092",
  "method": "bicubic",
  "parameters": {
    "lam": 0.25,
    "eps": 0.0,
    "gamma": 1.0,
    "dark_frac": 0.3
  }
}
```

## How It Works

### Image Size Relationship

The attack assumes a 4:1 scaling relationship.

If the uploaded decoy image is `4368x4368`, the target text image is generated at:

```text
4368 / 4 = 1092
```

The backend enforces:

- the decoy image must be square
- width and height must match
- width must be divisible by 4

The target text image and returned downsampled preview are both `decoy_width / 4` pixels wide and high.

### Base64 Image Flow

The browser reads the uploaded decoy with `FileReader.readAsDataURL()`. It sends the data URL in JSON as `decoy_image`.

The backend decodes it in `base64_to_image()`:

1. Verifies the payload is a string.
2. Removes a `data:image/...;base64,` prefix if present.
3. Validates the base64 character set.
4. Decodes bytes with `base64.b64decode(..., validate=True)`.
5. Opens the image with Pillow.
6. Allows PNG, JPEG, JPG, BMP, and TIFF.
7. Converts the image to RGB.
8. Returns a NumPy array shaped as `(height, width, 3)`.

Responses use `image_to_base64()`, which converts a NumPy array to a PNG data URL.

### Text Rendering

Text rendering happens in `create_text_image()`.

The function creates an RGB square image with a dark background:

```text
#333333
```

It draws target text in green:

```text
#00b002
```

Font loading tries:

1. `Arial.ttf`
2. common system font paths
3. Pillow's default font

Text is wrapped by `wrap_text_to_fit()`. The wrapper:

- splits text into paragraphs using explicit newline characters
- wraps normal words by measuring rendered text width with `ImageDraw.textbbox()`
- splits a single long token if the token is wider than the image text area
- returns lines that fit within the target width

Alignment controls where the wrapped line block is drawn:

```text
center
top
bottom
left
right
top-left
top-right
bottom-left
bottom-right
```

`text_overflowed` becomes `true` when the total wrapped line height is greater than the drawable text area height.

### Generator Execution

`/generate-adversarial` writes two temporary files:

```text
decoy.png
target.png
```

Then it launches one of the generator scripts as a subprocess using the active Python interpreter:

```text
backend/adversarial_generators/nearest_gen_payload.py
backend/adversarial_generators/bicubic_gen_payload.py
backend/adversarial_generators/bilinear_gen_payload.py
```

The subprocess runs inside a temporary directory. The generator writes an adversarial PNG there. The Flask app then locates the output file, loads it, creates the downsampled preview, and returns all images as base64 data URLs.

Output file detection is method-specific:

```text
nearest  -> advNN*.png, excluding *_down.png
bicubic  -> adv*.png
bilinear -> adv_bilinear*.png
```

### Method Algorithms

All methods work by modifying a high-resolution decoy so that a specific 4:1 downsampling operation produces the target text image.

#### `nearest`

`nearest` assumes nearest-neighbor downsampling samples one pixel from each `4x4` source block.

The `offset` parameter selects the sampled pixel location inside each block. The default is `2`, matching the center-biased sampling behavior used by Pillow for the 4:1 shrink used here.

Parameters:

- `lam`: controls mean-preservation strength
- `eps`: optional null-space dither strength
- `gamma`: target intensity pre-emphasis
- `offset`: sampled source-pixel offset inside each `4x4` block

Preview downsampling for `nearest` uses TensorFlow nearest-neighbor resize.

#### `bicubic`

`bicubic` embeds the target against a bicubic downsampling model. It uses a dark-region mask so changes are concentrated in lower-luma areas of the decoy image.

Parameters:

- `lam`: controls mean-preservation strength
- `eps`: optional null-space dither strength
- `gamma`: target intensity pre-emphasis
- `dark_frac`: fraction of the observed luma range considered editable

Preview downsampling for `bicubic` uses OpenCV cubic interpolation:

```text
cv2.INTER_CUBIC
```

#### `bilinear`

`bilinear` embeds the target against a bilinear downsampling model. For each target pixel, the algorithm considers the corresponding `4x4` source block and uses a bilinear weight vector focused around the block center.

Like `bicubic`, it restricts edits to darker areas using linear-light Rec.709 luma.

Parameters:

- `lam`: controls mean-preservation strength
- `eps`: optional null-space dither strength
- `gamma`: target intensity pre-emphasis
- `dark_frac`: fraction of the observed luma range considered editable
- `anti_alias`: switches the bilinear verification/downsample path

When `anti_alias` is `false`, bilinear preview downsampling uses:

```text
cv2.INTER_LINEAR_EXACT
```

When `anti_alias` is `true`, it uses:

```text
cv2.INTER_LINEAR
```

### Downsampled Preview Mapping

The returned `downsampled_image` is created after generation by `downsample_adversarial_image()`.

Current mapping:

```text
nearest  -> TensorFlow nearest-neighbor
bicubic  -> OpenCV INTER_CUBIC
bilinear -> OpenCV INTER_LINEAR_EXACT or INTER_LINEAR
```

This preview is the image that demonstrates whether the hidden text appears after scaling.

### Frontend Behavior

The frontend has no build step. It uses plain DOM APIs and `fetch()`.

Important browser-side behavior:

- validates that text is present before preview/generation
- validates that an uploaded decoy is an image
- checks that the decoy is square
- checks that the decoy width is divisible by 4
- shows method-specific controls
- sends JSON to the backend
- renders `target_image`, `adversarial_image`, and `downsampled_image`

The frontend does not perform the adversarial algorithm. All image generation and downsampling are backend responsibilities.

## Validation And Error Handling

The backend returns `400` for user input problems:

- invalid JSON
- invalid base64
- missing decoy image
- unsupported image format
- non-square decoy
- decoy dimensions not divisible by 4
- unsupported method
- invalid alignment
- invalid numeric parameters
- text longer than the configured maximum

It returns `500` for runtime/generator failures:

- subprocess exits nonzero
- generator fails to produce an output image
- unexpected server exceptions

The global exception handler logs unexpected exceptions and returns:

```json
{
  "error": "An unexpected error occurred"
}
```

## Development Notes

The `anamorpher/` directory is kept as credited upstream reference material. The working implementation lives in the root project directories:

```text
backend/
frontend/
tests/
```

Do not edit `anamorpher/` when changing this project.

Useful commands:

```powershell
python -m pytest -q
uv run python backend/app.py
```

If the server is already running and you change backend files, stop and restart it so the new code is loaded.
