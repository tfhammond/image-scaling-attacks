from abc import ABC, abstractmethod

import numpy as np


class BaseDownsampler(ABC):
    """Base class for all downsamplers."""

    @abstractmethod
    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        """Downsample an image to the target size using the specified method."""

    @abstractmethod
    def get_supported_methods(self) -> list:
        """Return the supported interpolation methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the downsampler name."""
