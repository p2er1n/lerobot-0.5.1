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

"""Async inference robot client that sends camera frames as JPEG.

Example:
```shell
python -m lerobot.async_inference.compressed_robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=piper_follower \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="pick up the nut" \
    --server_address=127.0.0.1:8080 \
    --policy_type=pi05_crvla \
    --pretrained_name_or_path=/path/to/pretrained_model \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --jpeg_quality=8
```
"""

import logging
import pickle  # nosec
import threading
import time
from dataclasses import asdict
from pprint import pformat

import draccus
import grpc

from lerobot.transport import services_pb2
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import CompressedRobotClientConfig
from .helpers import TimedObservation, get_logger, visualize_action_queue_size
from .jpeg_transport import compress_timed_observation
from .robot_client import RobotClient


class CompressedRobotClient(RobotClient):
    """RobotClient variant that JPEG-compresses camera fields on the wire."""

    prefix = "compressed_robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: CompressedRobotClientConfig):
        super().__init__(config)
        self.config = config
        self.camera_keys = tuple(
            key
            for key, feature in self.robot.observation_features.items()
            if isinstance(feature, tuple) and len(feature) == 3
        )
        if not self.camera_keys:
            self.logger.warning("Robot exposes no HWC camera features; observations will not be compressed")
        else:
            self.logger.info(
                "JPEG transport enabled for cameras %s at quality %d",
                self.camera_keys,
                self.config.jpeg_quality,
            )

    def send_observation(self, obs: TimedObservation) -> bool:
        """JPEG-compress camera frames and send an otherwise standard observation."""
        if not self.running:
            raise RuntimeError(
                "Client not running. Run CompressedRobotClient.start() before sending observations."
            )
        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        transport_observation, compression_stats = compress_timed_observation(
            obs,
            self.camera_keys,
            self.config.jpeg_quality,
        )
        observation_bytes = pickle.dumps(transport_observation)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(
            "Compressed %d camera frame(s): %d -> %d bytes (%.1f%%); "
            "gRPC payload=%d bytes; serialization=%.3fms",
            compression_stats.camera_count,
            compression_stats.raw_image_bytes,
            compression_stats.encoded_image_bytes,
            compression_stats.ratio * 100,
            len(observation_bytes),
            serialize_time * 1000,
        )

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[COMPRESSED CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            self.logger.debug(f"Sent JPEG-compressed observation #{obs.get_timestep()}")
            return True
        except grpc.RpcError as error:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {error}")
            return False


@draccus.wrap()
def async_compressed_client(cfg: CompressedRobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = CompressedRobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_compressed_client()
