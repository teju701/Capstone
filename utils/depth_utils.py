import numpy as np

# Cityscapes camera calibration constants
FOCAL_LENGTH = 2262.52
BASELINE     = 0.209313

# Depth range to keep (in metres) — clips sky/far noise and very close objects
MIN_DEPTH = 1.0    # metres
MAX_DEPTH = 80.0   # metres


def disparity_to_depth(disparity: np.ndarray) -> np.ndarray:
    """
    Converts a Cityscapes 16-bit disparity PNG to metric depth in metres,
    then normalises to [0, 1] for stable training.

    Cityscapes encoding:
        real_disparity = (raw_value - 1) / 256    if raw_value > 0
        real_disparity = 0                         if raw_value == 0  (invalid)

    Depth (metres) = focal_length * baseline / real_disparity

    Returns:
        depth: float32 array, same H×W as input
               valid pixels  → normalised metric depth in (0, 1]
               invalid pixels → 0.0  (masked out during training)
    """
    disparity = disparity.astype(np.float32)

    depth = np.zeros_like(disparity)          # default: invalid = 0

    valid = disparity > 0
    real_disp = (disparity[valid] - 1.0) / 256.0

    # Avoid divide-by-zero for near-zero disparities
    real_disp = np.clip(real_disp, 1e-3, None)

    metric_depth = (FOCAL_LENGTH * BASELINE) / real_disp

    # Clip to a sensible range and keep only valid metric depths
    in_range = (metric_depth >= MIN_DEPTH) & (metric_depth <= MAX_DEPTH)

    # Build output: normalise to [0, 1] using the fixed depth range
    # so sigmoid output in [0,1] maps directly to [MIN_DEPTH, MAX_DEPTH]
    depth_normed = (metric_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)

    # Only write pixels that are valid AND in metric range
    idx = np.where(valid)[0]
    # valid is a flat mask over the array, so we need to handle 2D properly
    valid_indices        = np.zeros_like(depth, dtype=bool)
    valid_indices[valid] = in_range
    depth[valid_indices] = depth_normed[in_range]

    return depth.astype(np.float32)