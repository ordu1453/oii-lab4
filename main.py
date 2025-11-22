import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------------
# 1. Датасет
transform = transforms.ToTensor()
full_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)

number_of_epochs = 3

# ------------------------------
# 2. Сеть с 1 скрытым слоем
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
    

# 2. Сеть с 1 скрытым слоем
class Net0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
    
class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# ------------------------------
# 3. Функции
def to_one_hot(y, num_classes=10):
    y_onehot = torch.zeros(len(y), num_classes)
    y_onehot[torch.arange(len(y)), y] = 1
    return y_onehot

def train_epoch(model, train_loader, optimizer, loss_fn, train_losses, train_accs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        y_onehot = to_one_hot(y)
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y_onehot)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = preds.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    epoch_loss = total_loss
    epoch_acc = correct / total

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    return epoch_loss, epoch_acc

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

def split_dataset(dataset, train_ratio):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set

# ------------------------------
# 4. Основной цикл по долям обучающей выборки
splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_accuracies = []
test_accuracies = []

for train_ratio in splits:
    train_set, test_set = split_dataset(full_data, train_ratio)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = Net0()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    # записываем историю для графика внутри train_epoch
    train_losses = []
    train_accs = []

    # 5 эпох обучения
    for epoch in range(number_of_epochs):
        train_epoch(model, train_loader, optimizer, loss_fn, train_losses, train_accs)

    # оценка на train и test
    train_acc = evaluate(model, train_loader)
    test_acc  = evaluate(model, test_loader)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"{train_ratio} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

# ------------------------------
# 5. Построение графика
plt.figure(figsize=(10,6))
plt.plot(splits, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(splits, test_accuracies, marker='s', label='Test Accuracy')
plt.xlabel('Доля обучающей выборки')
plt.ylabel('Accuracy')
plt.title('Train/Test Accuracy vs Доля обучающей выборки')
plt.grid(True)
plt.legend()
plt.show()
