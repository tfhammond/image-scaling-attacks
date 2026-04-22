import cv2
import numpy as np

from .base import BaseDownsampler


class OpenCVDownsampler(BaseDownsampler):
    """OpenCV-based downsampler adapted from the Anamorpher project backend."""

    def __init__(self):
        self._method_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }

    @property
    def name(self) -> str:
        return "OpenCV"

    def get_supported_methods(self) -> list:
        return list(self._method_map.keys())

    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")

        cv_method = self._method_map[method]
        return cv2.resize(image, target_size, interpolation=cv_method)

    def downsample_bilinear(self, image: np.ndarray, target_size: tuple, anti_alias: bool = False) -> np.ndarray:
        interpolation = cv2.INTER_LINEAR if anti_alias else cv2.INTER_LINEAR_EXACT
        return cv2.resize(image, target_size, interpolation=interpolation)
