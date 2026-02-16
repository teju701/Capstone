import numpy as np

# Cityscapes official mapping
ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

IGNORE_LABEL = 255

def encode_segmap(mask):
    lut = np.ones(256, dtype=np.uint8) * 255
    for k, v in ID_TO_TRAINID.items():
        lut[k] = v
    return lut[mask]
