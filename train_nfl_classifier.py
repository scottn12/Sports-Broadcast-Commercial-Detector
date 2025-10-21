import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# 1. Dataset class
class NFLDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Expects directory structure:
        image_dir/
            game/
                img1.jpg
                img2.jpg
            commercial/
                img1.jpg
                img2.jpg
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Load game images (label = 1)
        game_dir = os.path.join(image_dir, "game")
        if os.path.exists(game_dir):
            for img_name in os.listdir(game_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(game_dir, img_name))
                    self.labels.append(1)

        # Load commercial images (label = 0)
        commercial_dir = os.path.join(image_dir, "commercial")
        if os.path.exists(commercial_dir):
            for img_name in os.listdir(commercial_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(commercial_dir, img_name))
                    self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 2. Data transformations
# This is done for two reasons
# 1) ResNet-50 expects 224x224 input and ImageNet normalization. This is required to use the pre-trained model.
# 2) Data augmentation to improve generalization. Changing the image slightly (i.e. rotation, color) helps make sure the model
#    is not just memorizing the training images.
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# 3. Load pre-trained model and modify for binary classification
def create_model():
    model = models.resnet50(pretrained=True)

    # Freeze early layers (optional - speeds up training)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    return model


# 4. Training function
def train_model(model, train_loader, val_loader, epochs=10, device="cuda"):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}:")
        print(
            f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%"
        )
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "nfl_classifier_best.pth")
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")

        print()

    return model


# 5. Testing function with detailed metrics
def test_model(model, test_loader, device="cuda"):
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()

            # Handle single item batches
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            predicted = (outputs > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(outputs.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)

    print("=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"\nOverall Accuracy: {accuracy:.2f}%\n")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Commercial  Game")
    print(f"Actual Commercial    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"Actual Game          {cm[1][0]:4d}      {cm[1][1]:4d}\n")

    # Classification Report
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Commercial", "NFL Game"],
            digits=3,
        )
    )

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"NFL Game Detection:")
    print(f"  Precision: {precision:.3f} (of predicted games, how many were correct)")
    print(f"  Recall: {recall:.3f} (of actual games, how many were detected)")
    print(f"  F1-Score: {f1:.3f}\n")

    return accuracy, all_predictions, all_labels, all_probabilities


# 6. Inference function
def predict_image(image_path, model, transform, device="cuda"):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).item()

    label = "NFL Game" if output > 0.5 else "Commercial"
    confidence = output if output > 0.5 else (1 - output)

    return label, confidence


# 7. Main execution
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create datasets
    # NOTE: You need to organize your images in this structure:
    # data/
    #   train/
    #     game/
    #     commercial/
    #   val/
    #     game/
    #     commercial/
    #   test/
    #     game/
    #     commercial/

    train_dataset = NFLDataset("data/train", transform=train_transform)
    val_dataset = NFLDataset("data/val", transform=test_transform)
    test_dataset = NFLDataset("data/test", transform=test_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create and train model
    model = create_model()
    print("Training model...\n")
    model = train_model(model, train_loader, val_loader, epochs=10, device=device)

    # Load best model for testing
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load("nfl_classifier_best.pth"))

    # Test the model
    test_accuracy, predictions, labels, probabilities = test_model(
        model, test_loader, device
    )

    # Example: Predict a single image
    # accuracy, confidence = predict_image('path/to/image.jpg', model, test_transform, device)
    # print(f"Prediction: {accuracy} (confidence: {confidence:.2%})")
