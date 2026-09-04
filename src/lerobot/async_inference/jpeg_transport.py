# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JPEG transport objects for compressed asynchronous observations.

``JPEGTransportImage`` serializes as JPEG bytes but reconstructs as an RGB NumPy
array during unpickling. Consequently, an unchanged ``PolicyServer`` receives a
normal ``TimedObservation`` while the gRPC payload carries compressed images.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .helpers import TimedObservation


def _decode_jpeg_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as compressed:
        return np.asarray(compressed.convert("RGB")).copy()


@dataclass(frozen=True)
class JPEGTransportImage:
    """A picklable JPEG frame that becomes an RGB array when unpickled."""

    data: bytes

    def __reduce__(self):
        return _decode_jpeg_image, (self.data,)


@dataclass(frozen=True)
class JPEGCompressionStats:
    camera_count: int
    raw_image_bytes: int
    encoded_image_bytes: int

    @property
    def ratio(self) -> float:
        if self.raw_image_bytes == 0:
            return 1.0
        return self.encoded_image_bytes / self.raw_image_bytes


def encode_jpeg_image(image: np.ndarray, quality: int) -> JPEGTransportImage:
    """Encode one HWC uint8 RGB camera frame using Pillow's JPEG encoder."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Camera frames must be NumPy arrays, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"Camera frames must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera frames must have shape (H, W, 3), got {image.shape}")
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return JPEGTransportImage(buffer.getvalue())


def compress_timed_observation(
    observation: TimedObservation,
    camera_keys: Iterable[str],
    quality: int,
) -> tuple[TimedObservation, JPEGCompressionStats]:
    """Return a transport copy with only the configured camera fields compressed."""
    transport_data = observation.get_observation().copy()
    raw_image_bytes = 0
    encoded_image_bytes = 0
    camera_count = 0

    for camera_key in camera_keys:
        if camera_key not in transport_data:
            continue
        image = transport_data[camera_key]
        encoded_image = encode_jpeg_image(image, quality)
        transport_data[camera_key] = encoded_image
        raw_image_bytes += image.nbytes
        encoded_image_bytes += len(encoded_image.data)
        camera_count += 1

    transport_observation = replace(observation, observation=transport_data)
    stats = JPEGCompressionStats(camera_count, raw_image_bytes, encoded_image_bytes)
    return transport_observation, stats
