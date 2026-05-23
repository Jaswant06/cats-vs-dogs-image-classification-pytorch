```markdown
# Cats vs Dogs Image Classification Using PyTorch

## Project Goal

This project classifies cat and dog images using PyTorch.

The goal was to learn computer vision with deep learning by building two models:

1. A custom CNN trained from scratch
2. A transfer learning model using pretrained ResNet18

The project shows the difference between training a small CNN from scratch and using a pretrained model that already understands general image features.

## Dataset

The dataset is stored locally in the following structure:

```text
data/
+-- train/
|   +-- cats/
|   +-- dogs/
+-- val/
    +-- cats/
    +-- dogs/
```

Dataset used for training:

```text
Training images: 9,998
Validation images: 2,000
```

Classes:

```text
cats
dogs
```

The dataset was originally from the Kaggle Cats and Dogs image dataset.

## Tools Used

- Python
- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- Pillow

## Device Support

The training scripts support:

```text
MPS for Apple Silicon Macs
CUDA for NVIDIA GPUs
CPU fallback
```

On my machine, training used:

```text
MPS
```

## Image Preprocessing

For the custom CNN, images were resized to:

```text
128 x 128
```

Transforms used:

- resize
- random horizontal flip
- convert to tensor

For ResNet18 transfer learning, images were resized to:

```text
224 x 224
```

and normalized using ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## Part 1: Custom CNN From Scratch

The first model was a custom convolutional neural network.

Architecture:

```text
Conv2d: 3 -> 16
ReLU
MaxPool2d

Conv2d: 16 -> 32
ReLU
MaxPool2d

Conv2d: 32 -> 64
ReLU
MaxPool2d

Flatten
Linear
ReLU
Dropout
Linear: 2 output classes
```

This model learned from the cats and dogs dataset only.

## Custom CNN Results

The custom CNN improved during training, but performance was limited.

Final result:

```text
Validation Accuracy: 79.95%
```

Classification report:

```text
              precision    recall  f1-score   support

        cats       0.79      0.81      0.80      1000
        dogs       0.80      0.79      0.80      1000

    accuracy                           0.80      2000
```

The model was learning, but it still made many mistakes.

Example issues:

- confused some dogs as cats
- struggled with unusual poses
- struggled with confusing backgrounds
- showed some overfitting

Training accuracy reached around:

```text
86.69%
```

Validation accuracy reached around:

```text
79.95%
```

This gap showed that the custom CNN was learning the training data better than it generalized to unseen validation images.

## Why The Custom CNN Was Limited

The custom CNN had to learn image features from scratch.

That means it had to learn basic visual patterns such as:

- edges
- textures
- fur
- ears
- eyes
- shapes
- animal body structure

With only around 10,000 training images, the model improved but was not strong enough to generalize like a larger pretrained model.

## Part 2: Transfer Learning With ResNet18

To improve performance, I used transfer learning with ResNet18.

ResNet18 was pretrained on ImageNet, so it already learned useful image features from a much larger dataset.

Instead of training the whole network from scratch, I froze the pretrained layers and replaced the final classification layer.

Original ResNet18 final layer:

```text
1000 ImageNet classes
```

New final layer:

```text
2 classes: cats and dogs
```

This allowed the model to reuse general image knowledge and only learn the final cat/dog decision boundary.

## ResNet18 Transfer Learning Results

Transfer learning performed much better than the custom CNN.

Best validation accuracy:

```text
97.65%
```

Final validation accuracy:

```text
97.45%
```

Classification report:

```text
              precision    recall  f1-score   support

        cats       0.98      0.96      0.97      1000
        dogs       0.96      0.98      0.97      1000

    accuracy                           0.97      2000
```

The model was strong and balanced across both classes.

## Performance Comparison

| Model | Validation Accuracy |
|---|---:|
| Custom CNN from scratch | 79.95% |
| ResNet18 transfer learning | 97.45% |

Transfer learning improved validation accuracy by about:

```text
17.5 percentage points
```

This shows how powerful pretrained models are for computer vision tasks.

## Key Lesson

The custom CNN was useful for learning how convolutional neural networks work.

But transfer learning was much better for performance.

The main lesson:

```text
Training from scratch teaches the fundamentals.
Transfer learning gives stronger real-world results.
```

## Outputs

The project saves:

```text
models/cats_vs_dogs_cnn.pth
models/cats_vs_dogs_resnet18.pth
plots/training_curves.png
plots/confusion_matrix.png
plots/sample_predictions.png
plots/resnet18_training_curves.png
plots/resnet18_confusion_matrix.png
plots/resnet18_sample_predictions.png
```

The `.pth` files are generated locally and ignored by Git to avoid committing large binary model files.

## Prediction Scripts

The project includes two prediction scripts.

Custom CNN prediction:

```bash
python src/predict.py
```

ResNet18 prediction:

```bash
python src/predict_transfer.py
```

The ResNet18 prediction script loads the trained transfer learning model, predicts validation images, and saves a prediction grid.

Sample result:

```text
12 / 12 sample predictions correct
```

Most predictions had very high confidence.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the custom CNN:

```bash
python src/train.py
```

Run custom CNN predictions:

```bash
python src/predict.py
```

Train the ResNet18 transfer learning model:

```bash
python src/train_transfer.py
```

Run ResNet18 predictions:

```bash
python src/predict_transfer.py
```

## Project Structure

```text
cats-vs-dogs-image-classification-pytorch/
+-- data/
|   +-- train/
|   |   +-- cats/
|   |   +-- dogs/
|   +-- val/
|       +-- cats/
|       +-- dogs/
+-- models/
|   +-- cats_vs_dogs_cnn.pth
|   +-- cats_vs_dogs_resnet18.pth
+-- plots/
|   +-- confusion_matrix.png
|   +-- sample_predictions.png
|   +-- training_curves.png
|   +-- resnet18_confusion_matrix.png
|   +-- resnet18_sample_predictions.png
|   +-- resnet18_training_curves.png
+-- src/
|   +-- train.py
|   +-- predict.py
|   +-- train_transfer.py
|   +-- predict_transfer.py
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## What I Learned

- How to load image datasets with `ImageFolder`
- How to use image transforms in `torchvision`
- How CNN layers learn visual patterns
- How `Conv2d`, `ReLU`, `MaxPool2d`, `Dropout`, and `Linear` layers work together
- How to train and validate a CNN
- How to evaluate image classifiers using precision, recall, F1-score, and confusion matrix
- Why a custom CNN can overfit on limited image data
- How transfer learning works
- Why pretrained ResNet18 performs much better than a small CNN trained from scratch
- How to save model weights and prediction plots