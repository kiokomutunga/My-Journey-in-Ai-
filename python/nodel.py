# train_transfer.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import os

# Config
data_dir = "dataset"
batch_size = 32
img_size = 224
num_epochs = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

# Transforms
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
val_ds = ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)
test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# Model: MobileNetV2
model = models.mobilenet_v2(pretrained=True)
# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

best_val_f1 = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_ds)

    # Validate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {epoch_loss:.4f}")
    report = classification_report(all_labels, all_preds, target_names=train_ds.classes, zero_division=0, output_dict=True)
    val_f1 = sum([report[c]['f1-score'] for c in train_ds.classes]) / num_classes
    print("Val F1 (avg):", val_f1)
    scheduler.step(epoch_loss)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model.")

# Final test evaluation
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=train_ds.classes))
