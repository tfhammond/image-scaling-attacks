# Image Scaling Attacks for Prompt Injection

## Background: Downscaling And Image Scaling

Image scaling is the process of resizing an image from one resolution to another, often through downsampling methods such as nearest-neighbor, bilinear, or bicubic interpolation. Because these methods combine and transform pixel information in different ways, the visual content of an image can change when it is resized. This project explores how those resizing behaviors can be exploited to create adversarial images that appear benign at full resolution but reveal hidden content after downsampling.

## Credits

https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems/

downsampling backend adapted from https://github.com/trailofbits/anamorpher

## What This Project Does

This project is a small web app for generating and previewing image scaling attacks.

The app lets a user:

1. Type hidden text.
2. Preview that text as a target image.
3. Upload a square decoy image.
4. Choose a downsampling attack method.
5. Generate an adversarial image.
6. View the full-size adversarial image and the downsampled image.

The goal is to make a high-resolution image that looks like the decoy at full size, but reveals the hidden text after it is downsampled.

## Setup

This project uses Python and Flask. The recommended way to install dependencies is with `uv`.

From the project root, run:

```powershell
uv sync
```

Then start the server:

```powershell
uv run python backend/app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

To run tests:

```powershell
python -m pytest -q
```

If the server is already running and you change backend files, stop it and start it again.

## Basic Project Structure

```text
backend/
  app.py
  sanitizer.py
  adversarial_generators/
  downsamplers/

frontend/
  index.html
  script.js
  styles.css

tests/
  test_backend_app.py
  test_bicubic_gen_payload.py
  test_nearest_gen_payload.py
```

The frontend is plain HTML, CSS, and JavaScript. There is no separate frontend build step.

The backend is a Flask app. It serves the frontend and handles image generation.

## How To Use The App

1. Enter the text you want to hide.
2. Choose a font size and alignment.
3. Click `Preview Text`.
4. Upload a square PNG image.
5. Choose `nearest`, `bicubic`, or `bilinear`.
6. Adjust the method parameters.
7. Click `Generate`.
8. Review the three output images:
   - target text image
   - full adversarial image
   - downsampled image

The uploaded decoy image must be a PNG, it must be square, and its width must be divisible by 4.
Larger images may provide more room for embedding, depending on the method and the decoy image content.

For example:

```text
4368x4368 -> 1092x1092
1024x1024 -> 256x256
```

## Basic Code Overview

The project has three main parts:

1. The frontend collects user input.
2. The backend validates the input and prepares image files.
3. A generator script creates the adversarial image.

The frontend sends requests to two main backend routes:

```text
POST /preview-text
POST /generate-adversarial
```

`/preview-text` creates an image from the user's text.

`/generate-adversarial` creates the full adversarial image and also returns a downsampled preview.

## How Image Generation Works

The backend receives the uploaded decoy image as base64 image data.

It checks that:

- the image is valid
- the image is square
- the image width is divisible by 4

Then it creates a target text image that is one fourth the width and height of the decoy.

For a `4368x4368` decoy image, the target image is:

```text
1092x1092
```

The backend writes two temporary files:

```text
decoy.png
target.png
```

Then it runs one generator script depending on the selected method:

```text
nearest
bicubic
bilinear
```

The generator modifies the high-resolution decoy image so that the hidden target text appears after downsampling.

## Downsampling Methods

Each method is tied to a specific downsampling behavior.

### Nearest

`nearest` uses nearest-neighbor scaling.

Nearest-neighbor downsampling chooses a source pixel from each block of pixels. In this project, the generated image is designed around a 4-to-1 size reduction.

The preview downsampling backend for `nearest` uses TensorFlow nearest-neighbor resizing.

Main parameters:

- `lam`: controls how strongly the image tries to preserve the decoy
- `eps`: adds optional small variation
- `gamma`: adjusts the target image intensity before embedding
- `offset`: controls which pixel position is sampled inside each block

### Bicubic

`bicubic` uses cubic interpolation.

Bicubic downsampling combines nearby pixels using a smoother interpolation curve. This can reveal information that is not obvious in the full-size image.

The preview downsampling backend for `bicubic` uses OpenCV:

```text
cv2.INTER_CUBIC
```

Main parameters:

- `lam`: controls how strongly the image tries to preserve the decoy
- `eps`: adds optional small variation
- `gamma`: adjusts the target image intensity before embedding
- `dark_frac`: controls how much of the darker part of the image can be changed

### Bilinear

`bilinear` uses linear interpolation.

Bilinear downsampling blends nearby pixels using a simpler interpolation method than bicubic.

The preview downsampling backend for `bilinear` uses OpenCV.

By default, it uses:

```text
cv2.INTER_LINEAR_EXACT
```

If anti-aliasing is enabled, it uses:

```text
cv2.INTER_LINEAR
```

Main parameters:

- `lam`: controls how strongly the image tries to preserve the decoy
- `eps`: adds optional small variation
- `gamma`: adjusts the target image intensity before embedding
- `dark_frac`: controls how much of the darker part of the image can be changed
- `anti_alias`: switches the bilinear preview mode

## API Summary

### `POST /preview-text`

Request:

```json
{
  "text": "Hidden text",
  "font_size": 32,
  "alignment": "center"
}
```

Response:

```json
{
  "image": "data:image/png;base64,...",
  "text_overflowed": false,
  "size": "1092x1092"
}
```

### `POST /generate-adversarial`

Request:

```json
{
  "decoy_image": "data:image/png;base64,...",
  "text": "Hidden text",
  "method": "bicubic",
  "font_size": 32,
  "alignment": "center",
  "lam": 0.25,
  "eps": 0.0,
  "gamma": 1.0,
  "dark_frac": 0.3
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

## Notes

- The app runs locally.
- The frontend and backend are served from the same Flask server.
- Uploaded images are sent to the backend as base64 data.
- The generated output is returned directly to the browser as base64 images.
- Temporary processing files are created during generation and removed afterward.
