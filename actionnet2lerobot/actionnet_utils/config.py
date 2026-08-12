"""Feature layout for the Fourier ActionNet dataset.

This follows the schema the delivered dataset uses (the copy under
``/data/taeyoung/data/vla_pretrain_dataset/action_net``), not the reference
converter published with the dataset. The two differ substantially, and it is the
delivered one that RLDX-1 training consumes.

The conversion code that produced it no longer exists -- ALIN Lab confirmed on
2026-07-29 that preprocessing was done ad-hoc and was not kept -- so the layout
below was recovered by comparing ``observation.state`` against the
``observation.robot_joints`` / ``observation.hand_joints`` columns of a delivered
parquet, and cross-checked against ``neural_gr1``'s ``modality.json``, which names
the same eight blocks at the same offsets.
"""

# --- source layout -----------------------------------------------------------
#
# state/robot is (n, 32) in the GR1's own joint order, and state/hand is (n, 12),
# six per hand, left first.

ROBOT_JOINTS = 32
HAND_JOINTS = 12
SOURCE_WIDTH = ROBOT_JOINTS + HAND_JOINTS

_LEFT_LEG = range(0, 6)
_RIGHT_LEG = range(6, 12)
_WAIST = range(12, 15)
_NECK = range(15, 18)
_LEFT_ARM = range(18, 25)
_RIGHT_ARM = range(25, 32)
_LEFT_HAND = range(ROBOT_JOINTS, ROBOT_JOINTS + 6)
_RIGHT_HAND = range(ROBOT_JOINTS + 6, SOURCE_WIDTH)

# --- target layout -----------------------------------------------------------
#
# Grouped by body part, left side before right. The legs never move during
# teleoperated manipulation and are all zeros, but they keep their slots so the
# vector is the GR1's whole body.

GR1_BLOCKS: tuple[tuple[str, range], ...] = (
    ("left_arm", _LEFT_ARM),
    ("left_hand", _LEFT_HAND),
    ("left_leg", _LEFT_LEG),
    ("neck", _NECK),
    ("right_arm", _RIGHT_ARM),
    ("right_hand", _RIGHT_HAND),
    ("right_leg", _RIGHT_LEG),
    ("waist", _WAIST),
)

# column i of the emitted vector takes column PERMUTATION[i] of [robot | hand]
PERMUTATION: tuple[int, ...] = tuple(
    index for _, block in GR1_BLOCKS for index in block
)
STATE_WIDTH = len(PERMUTATION)

assert STATE_WIDTH == SOURCE_WIDTH, "the permutation must be a bijection"
assert sorted(PERMUTATION) == list(range(SOURCE_WIDTH)), "every source column once"


def block_ranges() -> dict[str, tuple[int, int]]:
    """``{"left_arm": (0, 7), "left_hand": (7, 13), ...}`` over the emitted vector."""
    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for name, block in GR1_BLOCKS:
        ranges[name] = (start, start + len(block))
        start += len(block)
    return ranges


# --- LeRobot features --------------------------------------------------------

CAMERA_NAME = "primary"
VIDEO_KEY = f"observation.images.{CAMERA_NAME}"
RGB_SHAPE = (800, 1280, 3)
FPS = 30
ROBOT_TYPE = "ActionNet"


def motor_names(count: int) -> list[str]:
    """The delivered datasets name motors positionally rather than by joint."""
    return [f"m{index}" for index in range(count)]


def _vector(count: int) -> dict:
    return {
        "dtype": "float32",
        "shape": (count,),
        "names": {"motors": motor_names(count)},
    }


def generate_features() -> dict:
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": RGB_SHAPE,
            "names": ["height", "width", "rgb"],
        },
        "observation.state": _vector(STATE_WIDTH),
        "action": _vector(STATE_WIDTH),
        # the raw source vectors, kept unpermuted alongside the assembled ones
        "observation.robot_joints": _vector(ROBOT_JOINTS),
        "observation.hand_joints": _vector(HAND_JOINTS),
        "action.robot_joints": _vector(ROBOT_JOINTS),
        "action.hand_joints": _vector(HAND_JOINTS),
        # actions are absolute joint targets, so this is a copy of `action`; it
        # exists because the training config distinguishes absolute from delta
        "absolute_action": _vector(STATE_WIDTH),
    }


def generate_modality() -> dict:
    """``meta/modality.json`` -- the GR00T-style view the training stack reads.

    The delivered ActionNet dataset exposes state and action as one flat block,
    even though the columns are in the body-part order above. Splitting them into
    the eight named blocks (as ``neural_gr1`` does) would not match
    ``rldx/configs/data/pt_data_config.py``, which asks ActionNet for
    ``modality_keys=["state"]``.
    """
    return {
        "state": {"state": {"start": 0, "end": STATE_WIDTH}},
        "action": {"action": {"start": 0, "end": STATE_WIDTH, "absolute": True}},
        "video": {CAMERA_NAME: {"original_key": VIDEO_KEY}},
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
            "human.action.task_name": {},
            "human.validity": {},
        },
    }
