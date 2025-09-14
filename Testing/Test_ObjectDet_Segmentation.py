# ---------------------------
# STEP 1: Imports
# ---------------------------
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# STEP 2: Load Dataset (Oxford-IIIT Pet)
# ---------------------------
transform = T.Compose([
    T.Resize((128, 128)),   # smaller images for faster training
    T.ToTensor()
])

target_transform = T.Compose([
    T.Resize((128, 128)),
    T.PILToTensor()
])

dataset = torchvision.datasets.OxfordIIITPet(
    root="./data", download=True,
    target_types="segmentation",
    transform=transform,
    target_transform=target_transform
)

# Use only a small subset for speed
train_size = 200
val_size = 50
small_train, _ = random_split(dataset, [train_size, len(dataset) - train_size])
small_val, _ = random_split(dataset, [val_size, len(dataset) - val_size])

train_loader = DataLoader(small_train, batch_size=4, shuffle=True)
val_loader = DataLoader(small_val, batch_size=4)

# ---------------------------
# STEP 3: Model
# ---------------------------
from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(pretrained=True)
# Replace final classifier for 4 classes
model.classifier[4] = torch.nn.Conv2d(512, 4, kernel_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# STEP 4: Training Loop (1 epoch only)
# ---------------------------
for epoch in range(1):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.squeeze(1).long().to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/1], Loss: {total_loss/len(train_loader):.4f}")

# ---------------------------
# STEP 5: Visualize Predictions
# ---------------------------
def decode_segmap(image, nc=4):
    label_colors = np.array([
        (0, 0, 0),       # background
        (255, 0, 0),     # pet
        (0, 255, 0),     # outline
        (0, 0, 255),     # extra class
    ])
    r, g, b = np.zeros_like(image), np.zeros_like(image), np.zeros_like(image)
    for l in range(0, nc):
        idx = image == l
        r[idx], g[idx], b[idx] = label_colors[l]
    return np.stack([r, g, b], axis=2)

model.eval()
with torch.no_grad():
    images, targets = next(iter(val_loader))
    images = images.to(device)
    outputs = model(images)["out"]
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(3, 4, i+5)
        plt.imshow(decode_segmap(targets[i].squeeze().numpy()))
        plt.title("Mask (GT)")
        plt.axis("off")

        plt.subplot(3, 4, i+9)
        plt.imshow(decode_segmap(preds[i]))
        plt.title("Prediction")
        plt.axis("off")

    plt.show()
