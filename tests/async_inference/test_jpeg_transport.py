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

from __future__ import annotations

import pickle  # nosec

import numpy as np
import pytest

from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.jpeg_transport import compress_timed_observation, encode_jpeg_image


def _test_image(height: int = 96, width: int = 128) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=-1).astype(np.uint8)


def test_compressed_observation_unpickles_as_standard_timed_observation():
    image = _test_image()
    original = TimedObservation(
        timestamp=123.0,
        timestep=7,
        must_go=True,
        observation={"front": image, "joint.pos": 1.25, "task": "pick up the nut"},
    )

    transport_observation, stats = compress_timed_observation(original, ("front",), quality=8)
    compressed_payload = pickle.dumps(transport_observation)  # nosec
    restored = pickle.loads(compressed_payload)  # nosec

    assert isinstance(restored, TimedObservation)
    assert restored.timestamp == original.timestamp
    assert restored.timestep == original.timestep
    assert restored.must_go is True
    assert restored.observation["joint.pos"] == original.observation["joint.pos"]
    assert restored.observation["task"] == original.observation["task"]
    assert isinstance(restored.observation["front"], np.ndarray)
    assert restored.observation["front"].shape == image.shape
    assert restored.observation["front"].dtype == np.uint8
    assert not np.array_equal(restored.observation["front"], image)
    assert np.array_equal(original.observation["front"], image)
    assert stats.camera_count == 1
    assert stats.raw_image_bytes == image.nbytes
    assert stats.encoded_image_bytes < image.nbytes
    assert len(compressed_payload) < len(pickle.dumps(original))  # nosec


def test_only_selected_camera_fields_are_compressed():
    front = _test_image()
    wrist = np.flip(front, axis=1).copy()
    original = TimedObservation(
        timestamp=123.0,
        timestep=7,
        observation={"front": front, "wrist": wrist},
    )

    transport_observation, stats = compress_timed_observation(original, ("front",), quality=16)
    restored = pickle.loads(pickle.dumps(transport_observation))  # nosec

    assert stats.camera_count == 1
    assert not np.array_equal(restored.observation["front"], front)
    assert np.array_equal(restored.observation["wrist"], wrist)


@pytest.mark.parametrize("quality", [0, 101])
def test_encode_jpeg_image_rejects_invalid_quality(quality: int):
    with pytest.raises(ValueError, match="between 1 and 100"):
        encode_jpeg_image(_test_image(), quality)


def test_encode_jpeg_image_requires_hwc_uint8_rgb():
    with pytest.raises(ValueError, match="dtype uint8"):
        encode_jpeg_image(_test_image().astype(np.float32), quality=8)
    with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
        encode_jpeg_image(_test_image().transpose(2, 0, 1), quality=8)
