import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denormalize(img):
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)

def visualize_sample(sample):
    img = sample["image"].permute(1, 2, 0).numpy()
    img = denormalize(img)

    seg = sample["seg"].numpy()
    depth = sample["depth"].numpy()

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(seg)
    plt.title("Segmentation")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(depth, cmap="viridis")
    plt.title("Depth")
    plt.axis("off")

    plt.show()
