import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.cityscapes_dataset import CityscapesDataset
from models.seg_model import SegmentationModel
from models.segformer_encoder import SegFormerEncoder
from models.segformer_decoder import SegFormerDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
BATCH_SIZE = 4
LR = 6e-5
EPOCHS = 50
# IMPORTANT: Save to your Drive path so it persists after disconnection
SAVE_PATH = "best_seg_model.pth" 
CHECKPOINT_PATH = "last_checkpoint.pth"

# ---------------------------
# mIoU
# ---------------------------
def compute_miou(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []

    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        if union == 0:
            continue
        ious.append(inter / union)

    return np.mean(ious) if len(ious) else 0.0

# ---------------------------
# Color map
# ---------------------------
CITYSCAPES_COLORS = np.array([
 (128,64,128),(244,35,232),(70,70,70),(102,102,156),
 (190,153,153),(153,153,153),(250,170,30),(220,220,0),
 (107,142,35),(152,251,152),(70,130,180),(220,20,60),
 (255,0,0),(0,0,142),(0,0,70),(0,60,100),
 (0,80,100),(0,0,230),(119,11,32)
])

def save_prediction(mask, epoch):
    mask = mask.cpu().numpy()
    colored = CITYSCAPES_COLORS[mask]
    plt.imsave(f"pred_epoch_{epoch}.png",
                colored.astype(np.uint8))

# ---------------------------
def main():
    train_ds = CityscapesDataset("data/cityscapes","train")
    val_ds   = CityscapesDataset("data/cityscapes","val")

    train_loader = DataLoader(train_ds,BATCH_SIZE,
                              shuffle=True,num_workers=2)

    val_loader = DataLoader(val_ds,1,
                            shuffle=False,num_workers=2)

    encoder = SegFormerEncoder()
    decoder = SegFormerDecoder(NUM_CLASSES)
    model = SegmentationModel(encoder,decoder,NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(),lr=LR)
    scheduler = CosineAnnealingLR(optimizer,T_max=EPOCHS)

    # --- RESUME LOGIC ---
    start_epoch = 0
    best_miou = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0)
        print(f"Resuming from Epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            img = batch["image"].to(DEVICE)
            gt  = batch["seg"].to(DEVICE)

            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out,gt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss/len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        miou_sum = 0

        with torch.no_grad():
            for i,batch in enumerate(val_loader):
                img = batch["image"].to(DEVICE)
                gt  = batch["seg"].to(DEVICE)
                out = model(img)
                pred = torch.argmax(out,dim=1)

                miou_sum += compute_miou(pred,gt,NUM_CLASSES)

                if i==0:
                    save_prediction(pred[0],epoch+1)

        miou = miou_sum/len(val_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val mIoU : {miou:.4f}")

        # Save Best Model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(),SAVE_PATH)
            print("Best model saved")
        
        # --- SAVE CHECKPOINT EVERY EPOCH ---
        # This saves the state so we can resume if Colab crashes
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
        }, CHECKPOINT_PATH)

    print("Training finished")
    print("Best mIoU:",best_miou)

if __name__=="__main__":
    main()