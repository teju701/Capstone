import numpy as np

FOCAL_LENGTH = 2262.52
BASELINE = 0.209313

def disparity_to_depth(disparity):
    """
    Converts Cityscapes disparity PNG to inverse depth (preferred).
    """
    disparity = disparity.astype(np.float32)

    # Cityscapes-specific conversion
    real_disparity = (disparity - 1.0) / 256.0

    depth = np.zeros_like(real_disparity)
    mask = real_disparity > 0

    # Convert to depth
    depth[mask] = (FOCAL_LENGTH * BASELINE) / real_disparity[mask]

    # OPTIONAL but recommended for progressive fusion:
    # Convert to inverse depth
    depth[mask] = 1.0 / depth[mask]

    # Clamp for numerical stability
    depth = np.clip(depth, 1e-4, 1.0)

    return depth
