# PI0.5 CR-VLA

This policy ports the architecture-level CR-VLA release to LeRobot's PI0.5 implementation.
It keeps PI0.5's flow-matching action expert and adds:

- CPE/CaVE extraction from the original `[0, 1]` camera tensors.
- CAM restoration of the projected SigLIP image tokens.
- Compression-prior Action Anchor attention on the PI0.5 action embeddings.
- An optional training-only clean/compressed visual-feature reconstruction objective.
- An optional removable codec/channel-parameter prediction head.

The CR-VLA modules are derived from `CR-VLA-anonymous-review-code` under the MIT license in
`CRVLA_LICENSE`. The surrounding PI0.5 implementation retains LeRobot's Apache-2.0 license.

## Base PI0.5 checkpoints

`PI05CRVLAPolicy.from_pretrained` accepts either a `pi05` or `pi05_crvla` checkpoint. Loading a
plain PI0.5 checkpoint uses non-strict loading: all original PI0.5 parameters are restored and the
new CR-VLA parameters retain their initialization. Those new parameters must be trained or loaded
from a PI0.5 CR-VLA checkpoint before compression-robust evaluation.

## Paired reconstruction training

Set `crvla_reconstruction_enabled=true` and add one clean tensor for every compressed camera tensor
in the training batch. By default, the clean key is the image key followed by `.clean`, for example:

```text
observation.images.image
observation.images.image.clean
```

The clean branch is stop-gradient. The compressed branch receives the reconstruction objective and
the normal PI0.5 flow-matching objective. Its coefficient follows the configured linear curriculum.

## Channel-head training

Set `crvla_channel_head_enabled=true` and provide the configured `crvla.codec_id` class target and
`crvla.channel_parameters` regression target. Set either output dimension to zero when that target
is not used. The channel head is not consulted during action inference.
