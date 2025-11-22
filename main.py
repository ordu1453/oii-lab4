import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------------
# 1. Датасет
transform = transforms.ToTensor()
full_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)

# ------------------------------
# 2. Сети
class Net0(nn.Module):  # без скрытых слоёв
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.model(x)

class Net1(nn.Module):  # 1 скрытый слой
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

class Net5(nn.Module):  # 5 скрытых слоёв
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
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

def train_epoch(model, train_loader, optimizer, loss_fn):
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
    return total_loss, correct/total

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
    return correct/total

def split_dataset(dataset, train_ratio):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def required_sample_size(p, epsilon, confidence):
    """
    p - ожидаемая точность (accuracy)
    epsilon - допустимая ошибка
    confidence - вероятность успешного выполнения (например, 0.95)
    """
    delta = 1 - confidence
    sigma2 = p * (1 - p)
    N = sigma2 / (delta * epsilon**2)
    return int(N)

# ------------------------------
# 4. Основной цикл
splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
models = {'Net0': Net0, 'Net1': Net1, 'Net5': Net5}
colors = {'Net0':'r', 'Net1':'g', 'Net5':'b'}

plt.figure(figsize=(10,6))

# Параметры для Чебышёва
epsilon = 0.05      # допустимая ошибка
confidence = 0.95   # уверенность 95%

for name, model_class in models.items():
    train_accuracies = []
    test_accuracies = []
    chebyshev_sizes = []

    for train_ratio in splits:
        train_set, test_set = split_dataset(full_data, train_ratio)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_set, batch_size=64)

        model = model_class()  # новая сеть с нуля
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.KLDivLoss(reduction="batchmean")

        # Обучение 5 эпох
        for epoch in range(5):
            train_epoch(model, train_loader, optimizer, loss_fn)

        # Оценка точности
        train_acc = evaluate(model, train_loader)
        test_acc  = evaluate(model, test_loader)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Расчет минимального размера выборки по Чебышёву
        N_needed = required_sample_size(train_acc, epsilon, confidence)
        chebyshev_sizes.append(N_needed)

        print(f"{name} | train_ratio={train_ratio} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | N_needed={N_needed}")

    # График train/test accuracy
    plt.plot(splits, train_accuracies, marker='o', color=colors[name], linestyle='-', label=f'{name} Train')
    plt.plot(splits, test_accuracies, marker='s', color=colors[name], linestyle='--', label=f'{name} Test')

plt.xlabel('Доля обучающей выборки')
plt.ylabel('Accuracy')
plt.title('Train/Test Accuracy vs Доля обучающей выборки')
plt.grid(True)
plt.legend()
plt.show()
