import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx2torch import convert
import onnxruntime as ort
import optuna


# Целевая функция Optuna
def objective(trial):
    # Range of hyperparameters to choose from Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    # Fixed hyperparameters for first training
    num_epochs = 4
    num_classes = 10

    # Создание модели
    model = convert(onnx.load('output_onnx/mnist-custom_1.onnx'))  # Загрузка модели из ONNX

    # Определение оптимизатора
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Specific for MNIST integrated into PyTorch
    DATA_PATH = 'mnist-data-path'
    MODEL_STORE_PATH = 'model-store-path'

    # Transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Обучение модели
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Запуск прямого прохода
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    # Тестирование модели
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100

    # Возврат точности как метрики для Optuna
    return accuracy


# Запуск Optuna с определением количества trials (испытаний)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # здесь желательно задавать >100 trials

# Вывод результатов
print(f"Лучшая точность: {study.best_value}")
print(f"Лучшие параметры: {study.best_params}")
'''
Итоги прогона 100 trials:
[I 2024-08-29 17:09:00,012] Trial 99 finished with value: 99.07000000000001 and parameters: 
    {'learning_rate': 0.0006358052101504033, 'batch_size': 164}. Best is trial 37 with value: 99.41.
Лучшая точность: 99.41
Лучшие параметры: {'learning_rate': 8.844388491045688e-05, 'batch_size': 101}
'''

# Обучение модели с лучшими параметрами
# (ДОДЕЛАТЬ ПОЛНОЦЕННОЕ ОБУЧЕНИЕ ПОДГРУЖЕННОЙ МОДЕЛИ onnx С ЛУЧШЕЙ КОНФИГУРАЦИЕЙ, ЗАТЕМ СОХРАНЕНИЕ В onnx)
#best_params = study.best_params
#learning_rate = best_params['learning_rate']
#batch_size = best_params['batch_size']
