import numpy as np
import tensorflow as tf

from .base import BaseDownsampler


class TensorFlowDownsampler(BaseDownsampler):
    """TensorFlow-based downsampler adapted from the Anamorpher project backend."""

    def __init__(self):
        self._method_map = {
            "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            "bilinear": tf.image.ResizeMethod.BILINEAR,
            "bicubic": tf.image.ResizeMethod.BICUBIC,
        }

    @property
    def name(self) -> str:
        return "TensorFlow"

    def get_supported_methods(self) -> list:
        return list(self._method_map.keys())

    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")

        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        tf_size = (target_size[1], target_size[0])
        tf_method = self._method_map[method]
        resized = tf.image.resize(tensor, tf_size, method=tf_method)
        return tf.squeeze(resized, 0).numpy().astype(np.uint8)
