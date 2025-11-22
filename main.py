import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


splits = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1)
]

def split_dataset(dataset, train_ratio):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set

class Net0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                 # 28x28 -> 784
            nn.Linear(784, 10),           # единственный слой
            nn.LogSoftmax(dim=1)          # для KLDivLoss
        )
    
    def forward(self, x):
        return self.model(x)
    
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                 # 28x28 -> 784
            nn.Linear(784, 128),           # единственный слой
            nn.ReLU(),  
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

model = Net0()
model1 = Net1()
print(model)
print(model1)

loss_fn = nn.KLDivLoss(reduction="batchmean")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

def to_one_hot(y, num_classes=10):
    y_onehot = torch.zeros(len(y), num_classes)
    y_onehot[torch.arange(len(y)), y] = 1
    return y_onehot


for epoch in range(5):
    model.train()
    total_loss = 0

    for x, y in train_loader:

        y_onehot = to_one_hot(y)

        optimizer.zero_grad()

        preds = model(x)

        loss = loss_fn(preds, y_onehot)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

    return correct / total

train_acc = test(model, train_loader)
test_acc = test(model, test_loader)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
