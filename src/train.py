# src/train.py - Training Layer
# Complete training pipeline with logging and early stopping

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import csv
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import get_dataloaders
from src.model import get_model, count_parameters

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
torch.manual_seed(config["training"]["seed"])


def save_training_log(log_data):
    """
    Saves training metrics to CSV file.
    This is our experiment tracking system!
    """
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    log_path = config["paths"]["training_log"]
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)


def plot_training_curves(train_losses, val_losses,
                          train_accs, val_accs):
    """
    Plots and saves training curves after training completes.
    Shows loss and accuracy over epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Results", fontsize=14, fontweight="bold")

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss",
                 color="#3b82f6", linewidth=2)
    axes[0].plot(val_losses,   label="Val Loss",
                 color="#ef4444", linewidth=2)
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(train_accs, label="Train Accuracy",
                 color="#22c55e", linewidth=2)
    axes[1].plot(val_accs,   label="Val Accuracy",
                 color="#f59e0b", linewidth=2)
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    plt.savefig("logs/training_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Training curves saved to logs/training_curves.png")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trains model for one epoch. Returns loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluates model on validation or test set."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train():
    """
    Main training function.
    Trains CNN with early stopping and saves best model.
    """

    print("=" * 55)
    print("MNIST Digit Recognizer - Training")
    print("=" * 55)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device         : {device}")

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize model
    model     = get_model().to(device)
    print(f"Model          : {config['model']['name']}")
    print(f"Parameters     : {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    # Learning rate scheduler
    # Reduces LR by 50% if val loss doesnt improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Training settings
    epochs           = config["training"]["epochs"]
    patience         = config["training"]["early_stopping_patience"]
    best_val_loss    = float("inf")
    patience_counter = 0

    # Track metrics
    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    # Create model directory
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - start_time

        # Print epoch summary
        print(f"\nEpoch [{epoch}/{epochs}] "
              f"({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | "
              f"Val Acc  : {val_acc:.2f}%")
        print(f"  LR        : "
              f"{optimizer.param_groups[0]['lr']:.6f}")

        # Save training log
        save_training_log({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "train_acc":  round(train_acc, 2),
            "val_acc":    round(val_acc, 2),
            "lr":         optimizer.param_groups[0]["lr"]
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc
            }, config["paths"]["best_model"])
            print(f"  Best model saved! Val Loss: "
                  f"{val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. "
                  f"Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}!")
            break

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        train_accs, val_accs
    )

    print("\n" + "=" * 55)
    print("Training Complete!")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Model saved   : "
          f"{config['paths']['best_model']}")
    print("=" * 55)


if __name__ == "__main__":
    train()