from __future__ import annotations

import argparse
import sys
from math import log10

import cv2
import numpy as np
import numpy.typing as npt

"""Payload generator adapted from the Anamorpher project backend."""

ImageF32 = npt.NDArray[np.float32]
VecF32 = npt.NDArray[np.float32]


def srgb2lin(x: ImageF32) -> ImageF32:
    x = x / 255.0
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return y.astype(np.float32)


def lin2srgb(y: ImageF32) -> ImageF32:
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)


def bilinear_kernel(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Linear (triangle) kernel for bilinear interpolation."""
    ax = np.abs(x)
    return np.where(ax <= 1, 1 - ax, 0.0)


def weight_vector_bilinear(scale: int = 4) -> VecF32:
    """
    Compute bilinear weights for a 2x2 region when downsampling by `scale`.
    For scale=4, OpenCV bilinear uses only the 2x2 pixels at the center.
    """
    weights = np.zeros((scale, scale), dtype=np.float32)
    center = (scale - 1) / 2.0

    for y in range(scale):
        for x in range(scale):
            dy = abs(y - center)
            dx = abs(x - center)
            if dy < 1.0 and dx < 1.0:
                weights[y, x] = (1.0 - dy) * (1.0 - dx)

    weights = weights / weights.sum()
    return weights.astype(np.float32).reshape(-1)


def luma_linear(img: ImageF32) -> npt.NDArray[np.float32]:
    """Rec.709 luma in linear-light."""
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(
        np.float32
    )


def bottom_luma_mask(img: ImageF32, frac: float = 0.3) -> npt.NDArray[np.bool_]:
    """
    Boolean mask where luma is within the bottom `frac` of the observed luma range.
    """
    y_values = luma_linear(img)
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    thresh = y_min + frac * (y_max - y_min)
    return thresh >= y_values


def embed_bilinear(
    decoy: ImageF32,
    target: ImageF32,
    lam: float = 0.25,
    eps: float = 0.0,
    gamma_target: float = 1.0,
    dark_frac: float = 0.3,
) -> ImageF32:
    """
    Adjust a high-res decoy so bilinear 4:1 downscale matches `target`,
    but only modify pixels whose luma lies in the bottom `dark_frac`
    of the image's observed luma range.
    """
    scale = 4
    w_full: VecF32 = weight_vector_bilinear(scale)

    editable_mask = bottom_luma_mask(decoy, frac=dark_frac)

    adv = decoy.copy()
    tgt = (target**gamma_target).astype(np.float32)

    height, width, _ = tgt.shape
    for j in range(height):
        for i in range(width):
            y0, x0 = j * scale, i * scale
            block = adv[y0 : y0 + scale, x0 : x0 + scale]
            block_mask = editable_mask[y0 : y0 + scale, x0 : x0 + scale]

            mask_flat = block_mask.reshape(-1)
            idx = np.flatnonzero(mask_flat)
            if idx.size == 0:
                continue

            for channel in (0,):
                y_cur = float((w_full * block[..., channel].reshape(-1)).sum())
                diff = float(tgt[j, i, channel] - y_cur)

                w_sub = w_full[idx]
                count = float(w_sub.size)
                sum_w_sub = float(w_sub.sum())
                w_norm2_sub = float(w_sub @ w_sub)

                denom = (count * w_norm2_sub + lam**2) - (sum_w_sub**2)
                if abs(denom) < 1e-12:
                    continue

                delta_sub = diff * (count * w_sub - lam * sum_w_sub) / denom

                if eps > 0.0 and w_sub.size >= 3:
                    c_sub = np.vstack([w_sub, np.ones_like(w_sub, dtype=np.float32)])
                    _, _, vh_sub = np.linalg.svd(c_sub, full_matrices=True)
                    b_sub = vh_sub[2:].astype(np.float32)
                    if b_sub.size > 0:
                        delta_sub = delta_sub + eps * (
                            b_sub.T @ np.random.randn(b_sub.shape[0])
                        ).astype(np.float32)

                delta_vec = np.zeros_like(w_full, dtype=np.float32)
                delta_vec[idx] = delta_sub.astype(np.float32)
                block[..., channel] = block[..., channel] + delta_vec.reshape(scale, scale)

            adv[y0 : y0 + scale, x0 : x0 + scale] = block

    return adv.astype(np.float32)


def mse_psnr(a: ImageF32, b: ImageF32) -> tuple[float, float]:
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * log10(1.0 / mse)
    return mse, psnr


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--decoy", required=True, help="decoy image (PNG/JPEG)")
    parser.add_argument("--target", required=True, help="target image (PNG/JPEG)")
    parser.add_argument("--lam", type=float, default=0.25, help="mean-preservation weight")
    parser.add_argument("--eps", type=float, default=0.0, help="null-space dither")
    parser.add_argument("--gamma", type=float, default=1.0, help="target gamma pre-emphasis")
    parser.add_argument(
        "--dark-frac",
        type=float,
        default=0.3,
        help="fraction of luma range considered embeddable (bottom part)",
    )
    parser.add_argument(
        "--anti-alias",
        action="store_true",
        help="use anti-aliased bilinear (INTER_LINEAR) instead of INTER_LINEAR_EXACT",
    )
    args = parser.parse_args()

    decoy_bgr = cv2.imread(args.decoy, cv2.IMREAD_COLOR)
    if decoy_bgr is None:
        print(f"Error: Failed to read decoy image: {args.decoy}", file=sys.stderr)
        sys.exit(1)
    decoy_bgr = decoy_bgr.astype(np.float32)

    target_bgr = cv2.imread(args.target, cv2.IMREAD_COLOR)
    if target_bgr is None:
        print(f"Error: Failed to read target image: {args.target}", file=sys.stderr)
        sys.exit(1)
    target_bgr = target_bgr.astype(np.float32)

    decoy_srgb = cv2.cvtColor(decoy_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_srgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    decoy_lin = srgb2lin(decoy_srgb)
    target_lin = srgb2lin(target_srgb)

    adv_lin = embed_bilinear(
        decoy_lin,
        target_lin,
        lam=args.lam,
        eps=args.eps,
        gamma_target=args.gamma,
        dark_frac=args.dark_frac,
    )

    adv_srgb = lin2srgb(adv_lin).round().astype(np.uint8)
    adv_bgr = cv2.cvtColor(adv_srgb, cv2.COLOR_RGB2BGR)

    name_stub = f"adv_bilinear_{args.lam:g}_{args.eps:g}_{args.gamma:g}"
    cv2.imwrite(f"{name_stub}.png", adv_bgr)
    print(f"saved {name_stub}.png")

    interp_method = cv2.INTER_LINEAR if args.anti_alias else cv2.INTER_LINEAR_EXACT
    interp_name = "INTER_LINEAR" if args.anti_alias else "INTER_LINEAR_EXACT"

    downsampled_bgr = cv2.resize(
        adv_bgr, (target_srgb.shape[1], target_srgb.shape[0]), interpolation=interp_method
    )
    downsampled_rgb = cv2.cvtColor(downsampled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    downsampled_lin = srgb2lin(downsampled_rgb)

    mse, psnr = mse_psnr(target_lin, downsampled_lin)
    print(f"4x->1x OpenCV {interp_name} | MSE {mse:.6f} PSNR {psnr:.2f} dB")


if __name__ == "__main__":
    main()
