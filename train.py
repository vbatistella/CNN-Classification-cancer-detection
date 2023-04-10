from code.pre_process import pre_process
from code.dataset import CancerDataset
from code.model import CNN
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

def main():
    pre_process()
    # Set the directory path where the images are stored
    directory = 'data/'
    # Create the custom dataset
    cancer_dataset = CancerDataset(directory)

    # Calculate the sizes of the train, validation, and test sets
    dataset_size = len(cancer_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(cancer_dataset, [train_size, val_size, test_size])

    # Define the data loaders for the train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Create an instance of the CNN model
    model = CNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for 10 epochs with validation
    num_epochs = 40
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()

        # Train on train dataset
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on validation dataset
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

    # Evaluate on the test dataset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total}%")
    torch.save(model, "./cancer_classifier.pt")

if __name__ == "__main__":
    main()