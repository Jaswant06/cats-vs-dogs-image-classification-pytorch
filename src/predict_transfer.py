from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VAL_DIR = DATA_DIR / "val"
MODEL_PATH = BASE_DIR / "models" / "cats_vs_dogs_resnet18.pth"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
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

model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

images, labels = next(iter(val_loader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

print("\nResNet18 Sample Predictions")
print("---------------------------")

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

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

plt.figure(figsize=(12, 8))

for i in range(len(images_cpu)):
    plt.subplot(3, 4, i + 1)

    image = images_cpu[i] * std + mean
    image = image.permute(1, 2, 0).clamp(0, 1)

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
plt.savefig(PLOTS_DIR / "resnet18_sample_predictions.png", bbox_inches="tight")
plt.close()

print("\nSaved prediction plot: plots/resnet18_sample_predictions.png")