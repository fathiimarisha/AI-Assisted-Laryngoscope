import os, cv2, numpy as np
from PIL import Image

os.makedirs("images", exist_ok=True)
os.makedirs("masks", exist_ok=True)

# Create 5 dummy frames with simple circles as "glottis"
for i in range(5):
    # black background RGB image
    img = np.zeros((128,128,3), dtype=np.uint8)
    mask = np.zeros((128,128), dtype=np.uint8)

    # random circle
    center = (np.random.randint(40, 88), np.random.randint(40, 88))
    radius = np.random.randint(10, 20)
    cv2.circle(img, center, radius, (255,255,255), -1)   # bright glottis
    cv2.circle(mask, center, radius, 255, -1)            # binary mask

    # save
    Image.fromarray(img).save(f"images/frame{i+1}.jpg")
    Image.fromarray(mask).save(f"masks/frame{i+1}.png")

print("Sample dataset created: 5 frames in images/ and masks/")

import os, glob, cv2, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ======================
# STEP 1: Dataset Loader
# ======================
class GlottisDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files = sorted(glob.glob(img_dir + "/*.jpg"))
        self.mask_files = sorted(glob.glob(mask_dir + "/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[idx], 0)  # grayscale mask

        # Resize for small model
        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        # Convert to torch tensors
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, mask

# ======================
# STEP 2: Tiny U-Net
# ======================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv1 = DoubleConv(32, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bn = self.bottleneck(p2)

        u2 = self.up2(bn)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        return torch.sigmoid(self.final(c1))

# ======================
# STEP 3: Training
# ======================
# Use the dummy dataset you already created
train_dataset = GlottisDataset("images/", "masks/")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Train for a few epochs
for epoch in range(5):
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss={loss.item():.4f}")

# ======================
# STEP 4: Visualization
# ======================
imgs, masks = next(iter(train_loader))
with torch.no_grad():
    preds = model(imgs.to(device)).cpu()

i = 0
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(imgs[i].permute(1,2,0)); plt.title("Input")
plt.subplot(1,3,2); plt.imshow(masks[i][0], cmap="gray"); plt.title("GT Mask")
plt.subplot(1,3,3); plt.imshow(preds[i][0], cmap="gray"); plt.title("Pred Mask")
plt.show()

