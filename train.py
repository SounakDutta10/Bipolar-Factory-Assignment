import torch
import torch.optim as optim
from model.yolo_resnet import YOLOResNet
from utils.dataset_loader import YOLODataset
from torch.utils.data import DataLoader
from utils.transforms import get_transforms

def collate_fn(batch):
    # This function will handle variable-sized labels
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Stack images (assuming all images have the same size)
    images = torch.stack(images, 0)
    
    return images, labels  # Return labels as a list of different sizes

def train():
    # Automatically choose CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model to device
    model = YOLOResNet(num_classes=1).to(device)

    # Load dataset
    dataset = YOLODataset(
        image_dir="datasets/test.v4i.yolov5pytorch/train/images", 
        label_dir="datasets/test.v4i.yolov5pytorch/train/labels",  # Correct path for labels
        transforms=get_transforms()
    )
    
    # Use the custom collate_fn
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()  # Ensure model is in training mode

    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        epoch_loss = 0.0

        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            # labels will be a list of varying lengths, handle them as such
            # You need to implement your YOLO loss function accordingly, considering the varying number of labels

            # Forward pass
            preds = model(images)

            # Ensure preds requires gradient
            preds = preds.requires_grad_()  # Ensure preds require gradients

            # Replace this with your actual YOLO loss function
            loss = compute_yolo_loss(preds, labels)  # Define compute_yolo_loss function separately

            # Backprop
            optimizer.zero_grad()  # Zero the gradients before the backward pass
            loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model weights

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(loader)}")

    # Save model
    torch.save(model.state_dict(), "yolo_resnet.pth")
    print("Model saved to yolo_resnet.pth")

def compute_yolo_loss(preds, labels):
    # Implement your actual YOLO loss function here.
    # The following is a placeholder for the loss calculation.
    loss = torch.tensor(0.0, requires_grad=True)  # Replace with actual YOLO loss calculation
    return loss

if __name__ == "__main__":
    train()
