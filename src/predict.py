from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VAL_DIR = DATA_DIR / "val"
MODEL_PATH = BASE_DIR / "models" / "cats_vs_dogs_cnn.pth"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=12,
    shuffle=True,
)

class_names = val_dataset.classes

model = CatDogCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

images, labels = next(iter(val_loader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

print("\nSample Predictions")
print("------------------")

for i in range(len(images)):
    predicted_class = class_names[predictions[i].item()]
    actual_class = class_names[labels[i].item()]
    confidence = probabilities[i][predictions[i]].item()

    print(
        f"Image {i + 1}: "
        f"Predicted={predicted_class}, "
        f"Actual={actual_class}, "
        f"Confidence={confidence:.4f}"
    )

images_cpu = images.cpu()
labels_cpu = labels.cpu()
predictions_cpu = predictions.cpu()
probabilities_cpu = probabilities.cpu()

plt.figure(figsize=(12, 8))

for i in range(len(images_cpu)):
    plt.subplot(3, 4, i + 1)
    image = images_cpu[i].permute(1, 2, 0)
    plt.imshow(image)

    predicted_class = class_names[predictions_cpu[i].item()]
    actual_class = class_names[labels_cpu[i].item()]
    confidence = probabilities_cpu[i][predictions_cpu[i]].item()

    title_color = "green" if predicted_class == actual_class else "red"

    plt.title(
        f"Pred: {predicted_class}\n"
        f"Actual: {actual_class}\n"
        f"Conf: {confidence:.2f}",
        color=title_color,
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "sample_predictions.png", bbox_inches="tight")
plt.close()

print("\nSaved prediction plot: plots/sample_predictions.png")