import numpy as np
from PIL import Image


class PillowDownsampler:
    """Pillow-based downsampler adapted from the Anamorpher project backend."""

    def __init__(self):
        self._method_map = {
            "nearest": Image.Resampling.NEAREST,
            "bicubic": Image.Resampling.BICUBIC,
        }

    @property
    def name(self) -> str:
        return "Pillow"

    def get_supported_methods(self) -> list[str]:
        return list(self._method_map.keys())

    def downsample(self, image: np.ndarray, target_size: tuple[int, int], method: str) -> np.ndarray:
        """Downsample an image using Pillow resize with the selected method."""
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")

        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_pil = pil_image.resize(target_size, self._method_map[method])
        return np.array(resized_pil)
