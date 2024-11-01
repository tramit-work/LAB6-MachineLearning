import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    return model

def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='D:/Downloads/dataMLP', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1024, num_workers=8, shuffle=True
)

    testset = torchvision.datasets.MNIST(
        root='D:/Downloads/dataMLP', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=1024, num_workers=8, shuffle=False
)

    n_features = 28 * 28
    model = get_model(n_features)
    lr = 0.01
    optim = SGD(params=model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    return model, trainloader, testloader, loss_fn, optim

def evaluate(model, testloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss /= len(testloader)
    return test_loss, accuracy

if __name__ == '__main__':

    model, trainloader, testloader, loss_fn, optim = train_model()
    def imshow(img):
        img = img * 0.5 + 0.5
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    for i, (images, labels) in enumerate(trainloader, 0):
        imshow(torchvision.utils.make_grid(images[:8]))
        break
    n_epochs = 10
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(n_epochs):
        model.train()  
        running_loss = 0.0
        running_correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optim.zero_grad()  
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward() 
            optim.step()  

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            loss.backward()
            torch.optim.step()

        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / (i + 1)
        test_loss, test_accuracy = evaluate(model, testloader, loss_fn)

        print(f"Epoch [{epoch + 1}/{n_epochs}], "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('Loss over Epochs')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy over Epochs')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()

    plt.show()
    torch.save(model, "D:/Downloads/dataMLP/MLP_dress.pth")
