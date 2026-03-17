# src/evaluate.py - Evaluation Layer
# Confusion matrix, ANN vs CNN comparison, misclassified samples

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix,
                             classification_report)
import yaml
import os
from src.data import get_dataloaders, get_class_names
from src.model import get_model, get_ann_model, count_parameters

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_best_model():
    """Loads the best saved CNN model"""
    model     = get_model()
    checkpoint = torch.load(
        config["paths"]["best_model"],
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from : "
          f"{config['paths']['best_model']}")
    print(f"Saved at epoch    : "
          f"{checkpoint['epoch']}")
    print(f"Val Accuracy      : "
          f"{checkpoint['val_acc']:.2f}%")
    return model


def get_predictions(model, loader, device):
    """Gets all predictions and true labels for test set"""
    all_preds  = []
    all_labels = []
    all_images = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_images),
            np.array(all_probs))


def plot_confusion_matrix(labels, preds, class_names):
    """Plots and saves confusion matrix"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.title("Confusion Matrix - MNIST CNN",
              fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to logs/confusion_matrix.png")


def plot_misclassified(images, labels, preds,
                       probs, class_names, n=20):
    """Plots misclassified samples"""
    # Find misclassified indices
    wrong_idx = np.where(preds != labels)[0]

    if len(wrong_idx) == 0:
        print("No misclassified samples found!")
        return

    # Take first n misclassified
    wrong_idx = wrong_idx[:n]
    rows = 4
    cols = 5

    fig, axes = plt.subplots(rows, cols,
                              figsize=(15, 12))
    fig.suptitle(
        "Misclassified Samples",
        fontsize=14, fontweight="bold"
    )

    for i, idx in enumerate(wrong_idx):
        if i >= rows * cols:
            break
        ax = axes[i // cols][i % cols]

        # Show image
        img = images[idx][0]
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        confidence = probs[idx][preds[idx]] * 100

        ax.set_title(
            f"True: {true_label}\n"
            f"Pred: {pred_label} "
            f"({confidence:.1f}%)",
            fontsize=8,
            color="red"
        )

    plt.tight_layout()
    plt.savefig("logs/misclassified.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Misclassified samples saved to "
          "logs/misclassified.png")


def compare_ann_vs_cnn(test_loader, device):
    """
    Trains a simple ANN and compares with CNN.
    Shows why CNN is better for image data!
    """
    print("\nComparing ANN vs CNN...")
    print("Training ANN for 3 epochs (quick comparison)...")

    # Load CNN accuracy from saved model
    checkpoint = torch.load(
        config["paths"]["best_model"],
        map_location="cpu",
        weights_only=True
    )
    cnn_acc = checkpoint["val_acc"]

    # Train ANN quickly for comparison
    ann       = get_ann_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ann.parameters(), lr=0.001)

    # Quick 3 epoch training
    from src.data import get_dataloaders
    train_loader, val_loader, _ = get_dataloaders()

    ann.train()
    for epoch in range(3):
        correct = 0
        total   = 0
        for images, labels in train_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            optimizer.zero_grad()
            outputs = ann(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
        print(f"  ANN Epoch [{epoch+1}/3] "
              f"Train Acc: {100*correct/total:.2f}%")

    # Evaluate ANN on test set
    ann.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = ann(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    ann_acc = 100 * correct / total

    # Plot comparison
    models      = ["Simple ANN", "CNN (Ours)"]
    accuracies  = [ann_acc, cnn_acc]
    colors      = ["#ef4444", "#22c55e"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies,
                   color=colors, width=0.4,
                   edgecolor="white", linewidth=1.2)
    plt.title("ANN vs CNN Accuracy Comparison",
              fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.ylim(90, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{acc:.2f}%",
            ha="center",
            fontsize=12,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig("logs/ann_vs_cnn.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nANN Accuracy : {ann_acc:.2f}%")
    print(f"CNN Accuracy : {cnn_acc:.2f}%")
    print(f"CNN is better by : "
          f"{cnn_acc - ann_acc:.2f}%")
    print("ANN vs CNN chart saved to logs/ann_vs_cnn.png")


def evaluate():
    """Main evaluation function"""
    print("=" * 55)
    print("MNIST Digit Recognizer - Evaluation")
    print("=" * 55)

    device      = torch.device("cpu")
    class_names = get_class_names()

    # Load data
    _, _, test_loader = get_dataloaders()

    # Load best model
    print("\nLoading best model...")
    model = load_best_model()
    model = model.to(device)

    # Get predictions
    print("\nRunning predictions on test set...")
    preds, labels, images, probs = get_predictions(
        model, test_loader, device
    )

    # Calculate accuracy
    accuracy = (preds == labels).mean() * 100
    print(f"\nTest Accuracy : {accuracy:.2f}%")

    # Classification report
    print("\nPer-Class Accuracy:")
    print(classification_report(
        labels, preds,
        target_names=[f"Digit {i}" for i in range(10)]
    ))

    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(labels, preds, class_names)

    # Plot misclassified samples
    print("Finding misclassified samples...")
    plot_misclassified(
        images, labels, preds, probs, class_names
    )

    # ANN vs CNN comparison
    compare_ann_vs_cnn(test_loader, device)

    print("\n" + "=" * 55)
    print("Evaluation Complete!")
    print("All charts saved to logs/ folder!")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()