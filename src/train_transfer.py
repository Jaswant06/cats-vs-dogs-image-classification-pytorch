from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms,
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

print("Class names:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 10

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

final_predictions = []
final_labels = []

for epoch in range(epochs):
    model.train()

    running_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = train_correct / train_total

    model.eval()

    validation_loss = 0
    val_correct = 0
    val_total = 0

    epoch_predictions = []
    epoch_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            validation_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

            epoch_predictions.extend(predictions.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

    val_loss = validation_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    final_predictions = epoch_predictions
    final_labels = epoch_labels

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Train Loss: {train_loss:.4f} - "
        f"Train Acc: {train_accuracy:.4f} - "
        f"Val Loss: {val_loss:.4f} - "
        f"Val Acc: {val_accuracy:.4f}"
    )

torch.save(model.state_dict(), MODELS_DIR / "cats_vs_dogs_resnet18.pth")

print("Transfer learning model saved.")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker="o", label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, marker="o", label="Val Loss")
plt.title("ResNet18 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, marker="o", label="Train Accuracy")
plt.plot(range(1, epochs + 1), val_accuracies, marker="o", label="Val Accuracy")
plt.title("ResNet18 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "resnet18_training_curves.png", bbox_inches="tight")
plt.close()

print("Transfer learning curves saved.")

cm = confusion_matrix(final_labels, final_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes,
)
plt.title("ResNet18 Cats vs Dogs Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.savefig(PLOTS_DIR / "resnet18_confusion_matrix.png", bbox_inches="tight")
plt.close()

print("\nClassification Report:")
print(
    classification_report(
        final_labels,
        final_predictions,
        target_names=train_dataset.classes,
    )
)
print("Transfer learning confusion matrix saved.")