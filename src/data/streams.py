"""
Four-stream input derivation for MS-AAGCN-style fusion (Shi et al., IEEE TIP 2020).

Given the raw joint coordinates of a Penn Action sequence, derive any of:
    - joint        : raw (x, y) coordinates (identity)
    - bone         : directed difference joint - joint[parent]
    - joint_motion : frame-wise velocity of joints, zero-padded at t=T-1
    - bone_motion  : frame-wise velocity of bones, zero-padded at t=T-1

All four streams emit (T, V, C) tensors of identical shape, so the same
ST-GCN architecture consumes any of them unchanged.
"""

import numpy as np


# ─── Bone parent mapping ─────────────────────────────────────────────────────
# For each of the 13 joints, PENN_BONE_PARENT[j] is the index of joint j's
# parent in a spanning tree rooted at joint 7 (l_hip). The root has
# PARENT[7] = 7, so its bone vector is zero — this preserves the V=13 shape
# so the same ST-GCN architecture consumes every stream unchanged.
#
# All 12 parent-links correspond to actual skeleton edges from BONE_CONNECTIONS
# in src/data/preprocess.py. The one skeleton edge not used as a parent-link
# is (0, 2) — the cycle-breaker for the 0-1-7-8-2-0 cycle. That edge remains
# in the adjacency matrix A for spatial graph convolution; only the bone-
# derivation step excludes it.
PENN_BONE_PARENT = np.array([
    1,   # 0  head        <- 1  l_shoulder     (edge 0-1)
    7,   # 1  l_shoulder  <- 7  l_hip          (edge 1-7, spine)
    8,   # 2  r_shoulder  <- 8  r_hip          (edge 2-8, spine)
    1,   # 3  l_elbow     <- 1  l_shoulder     (edge 1-3)
    2,   # 4  r_elbow     <- 2  r_shoulder     (edge 2-4)
    3,   # 5  l_wrist     <- 3  l_elbow        (edge 3-5)
    4,   # 6  r_wrist     <- 4  r_elbow        (edge 4-6)
    7,   # 7  l_hip       <- self (root, zero bone)
    7,   # 8  r_hip       <- 7  l_hip          (edge 7-8, hip bridge)
    7,   # 9  l_knee      <- 7  l_hip          (edge 7-9)
    8,   # 10 r_knee      <- 8  r_hip          (edge 8-10)
    9,   # 11 l_ankle     <- 9  l_knee         (edge 9-11)
    10,  # 12 r_ankle     <- 10 r_knee         (edge 10-12)
], dtype=np.int64)


STREAMS = ('joint', 'bone', 'joint_motion', 'bone_motion')

# MS-AAGCN (Shi et al. IEEE TIP 2020) weights: (2:1:2:1). Joint and joint-motion
FUSION_WEIGHTS = {
    'joint':        2.0,
    'bone':         1.0,
    'joint_motion': 2.0,
    'bone_motion':  1.0,
}


def compute_bone_stream(joints: np.ndarray) -> np.ndarray:
    """
    Derive the bone stream from raw joint coordinates.

        bone[t, j, :] = joint[t, j, :] - joint[t, parent[j], :]

    The root joint (parent == self) produces the zero vector, for keeping the output shape at V = 13.

    Args:
        joints: (T, V, C) array of joint coordinates

    Returns:
        bones:  (T, V, C) array of bone vectors
    """
    return joints - joints[:, PENN_BONE_PARENT, :]


def compute_motion_stream(sequence: np.ndarray) -> np.ndarray:
    """
    Derive the motion (velocity) stream from a joint or bone sequence.

        motion[t, j, :] = sequence[t+1, j, :] - sequence[t, j, :]   for t < T-1
        motion[T-1, :, :] = 0                                        (zero-padded)

    The zero-pad at the final frame preserves T so all four streams have
    identical shape, matching MS-AAGCN's convention.

    Args:
        sequence: (T, V, C) array (joint coordinates or bone vectors)

    Returns:
        motion:   (T, V, C) array of frame differences
    """
    motion = np.zeros_like(sequence)
    motion[:-1] = sequence[1:] - sequence[:-1]
    return motion


def derive_stream(joints: np.ndarray, stream: str) -> np.ndarray:
    """
    Given raw joint coordinates, return the requested MS-AAGCN stream.

    Args:
        joints : (T, V, C) raw joint coordinates (post-augmentation)
        stream : one of {'joint', 'bone', 'joint_motion', 'bone_motion'}

    Returns:
        (T, V, C) tensor in the requested modality
    """
    if stream == 'joint':
        return joints
    if stream == 'bone':
        return compute_bone_stream(joints)
    if stream == 'joint_motion':
        return compute_motion_stream(joints)
    if stream == 'bone_motion':
        # Note: motion of bones, not bones of motion. These are equivalent
        # mathematically (both are second-order differences), but composing
        # as motion(bone(x)) matches MS-AAGCN's reference implementation.
        return compute_motion_stream(compute_bone_stream(joints))
    raise ValueError(
        f"Unknown stream '{stream}'. "
        f"Expected one of: {STREAMS}."
    )
