import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from dataset.cityscapes_dataset import CityscapesDataset # Your dataset file

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
BATCH_SIZE = 4
NUM_CLASSES = 19

def compute_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - intersection
    iou = intersection / (union + 1e-7)
    return iou

def validate():
    print(f"[1/4] Loading Model: {MODEL_NAME}")
    # We use the official class to ensure the decoder logic is perfect
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    print("[2/4] Initializing Validation Dataset...")
    val_dataset = CityscapesDataset(
        root="data/cityscapes", 
        split='val',            # Make sure this points to your val folder
        img_size=(512, 1024)
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Confusion matrix: [True, Pred]
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    print("[3/4] Running Inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch["image"].to(DEVICE)
            targets = batch["seg"].numpy() # Ground truth

            outputs = model(pixel_values=images)
            logits = outputs.logits

            # Upsample logits to match ground truth size (512, 1024)
            upsampled_logits = F.interpolate(
                logits, 
                size=targets.shape[1:], 
                mode='bilinear', 
                align_corners=False
            )
            
            preds = torch.argmax(upsampled_logits, dim=1).cpu().numpy()

            # Update Confusion Matrix
            for t, p in zip(targets.flatten(), preds.flatten()):
                if t == 255: # Ignore the void class
                    continue
                conf_matrix[t, p] += 1

    print("[4/4] Calculating Results...")
    ious = compute_iou(conf_matrix)
    
    # Cityscapes Class Names
    classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 
        'train', 'motorcycle', 'bicycle'
    ]

    print("\n--- Per Class IoU ---")
    for i, iou in enumerate(ious):
        print(f"{classes[i]:15s}: {iou*100:.2f}%")

    print(f"\n[FINAL RESULT] Mean IoU: {np.mean(ious)*100:.2f}%")

if __name__ == "__main__":
    validate()