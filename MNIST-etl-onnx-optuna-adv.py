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
import math
from torch.utils.data import random_split


# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Целевая функция Optuna
def objective(trial, path_to_onnx_model_optuna, number_epochs_optuna, criterion_optuna):
    # Range of hyperparameters to choose from Optuna (2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])  # при batch_size>total_step будет batch_size=total_step
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    adam_betas1 = trial.suggest_float('adam_betas1', 1e-1, 99e-2)
    adam_betas2 = trial.suggest_float('adam_betas2', 999e-3, 999999e-6)
    adam_eps = trial.suggest_float('adam_eps', 1e-10, 1e-6, log=True)
    adam_weight_decay = trial.suggest_float('adam_weight_decay', 1e-8, 1e-1, log=True)

    # Fixed hyperparameters needed for training
    num_epochs_optuna = number_epochs_optuna

    # Формирование массивов данных MNIST
    # Specific for MNIST integrated into PyTorch
    DATA_PATH = 'mnist-data-path'
    MODEL_STORE_PATH = 'model-store-path'
    # Transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST 70000 images dataset (60000 images for train, and 10000 images for test)
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    # Loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Задаем модель нейросети для Optuna (импорт из onnx)
    onnx_model = onnx.load(path_to_onnx_model_optuna)  # Загрузка модели из ONNX
    model = convert(onnx_model)  # Подготовка к дообучению
    model.to(device)  # Перенос модели на устройство GPU

    # Определение оптимизатора
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_betas1, adam_betas2), eps=adam_eps, weight_decay=adam_weight_decay)
    print('optimizer=', optimizer)

    # Loss
    criterion = criterion_optuna

    # Обучение модели для выбранной конфигурации гиперпараметров Optuna
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs_optuna):
        train_acc = 0
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)  # Перенос данных на устройство GPU
            labels = labels.to(device)  # Перенос меток на устройство GPU

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

            train_acc_step = (correct / total) * 100
            train_acc += train_acc_step
            train_acc_aver = train_acc / (i + 1)

        print(f"train_acc_aver: {train_acc_aver:.2f} %")
        trial.report(train_acc_aver, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Тестирование модели
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:

            images = images.to(device)  # Перенос данных на устройство GPU
            labels = labels.to(device)  # Перенос меток на устройство GPU

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = (correct / total) * 100

    # Возврат точности как метрики для Optuna
    return test_accuracy

# Ввод значений параметров и запуск Optuna
onnxpath = 'output_onnx/mnist-custom_single_run_1.onnx'
number_epochs_optuna = 10
# Loss
criterion = nn.CrossEntropyLoss()

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=50),
                            pruner=optuna.pruners.HyperbandPruner(),
                            direction='maximize')
study.optimize(lambda trial: objective(trial, onnxpath, number_epochs_optuna, criterion),
               n_trials=501)  # желательно задавать >100 trials

# Вывод результатов
print(f"Номер лучшей попытки: Trial {study.best_trial.number}")
print(f"Лучшая точность: {study.best_value}")
print(f"Лучшие параметры: {study.best_params}")
print(f"Количество обрезанных (pruned) trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

################################################################################################

# Обучение модели с лучшими параметрами
# Ввод прочих параметров
number_epochs_final = 20

# Ввод определенных Optuna лучших параметров
best_params = study.best_params
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
adam_betas1 = best_params['adam_betas1']
adam_betas2 = best_params['adam_betas2']
adam_eps = best_params['adam_eps']
adam_weight_decay = best_params['adam_weight_decay']

# Полноценное обучение с наилучшей комбинацией ГП от Optuna

# Формирование массивов данных MNIST
# Specific for MNIST integrated into PyTorch
DATA_PATH = 'mnist-data-path'
MODEL_STORE_PATH = 'model-store-path'
# Transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# MNIST 70000 images dataset (60000 images for train, and 10000 images for test)
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Подгрузка исходной модели onnx
onnx_model = onnx.load(onnxpath)
# Extract parameters from onnx into pytorch
torch_model = convert(onnx_model)
model = torch_model
model = model.to(device) # Перенос модели на устройство GPU
print('model=', model)

# Определение оптимизатора
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_betas1, adam_betas2), eps=adam_eps, weight_decay=adam_weight_decay)
print('optimizer=', optimizer)

# Обучение модели для выбранной конфигурации гиперпараметров Optuna
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(number_epochs_final):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на устройство
        labels = labels.to(device)  # Перенос меток на устройство

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

        if batch_size >= total_step and (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                .format(epoch + 1, number_epochs_final, i + 1, total_step,
                        total_step, loss.item(), (correct / total) * 100))
        elif (i + 1) % batch_size == 0:
            print('Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, number_epochs_final, i + 1, total_step, int((i + 1) / batch_size),
                          math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
        elif (i + 1) == total_step:
            print('Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, number_epochs_final, i + 1, total_step, (int((i + 1) / batch_size)) + 1,
                          math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))

# Test the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        images = images.to(device)  # Перенос данных на устройство GPU
        labels = labels.to(device)  # Перенос меток на устройство GPU

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save trained model into onnx
torch_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(
    model,  # PyTorch model
    (torch_input,),  # Input data
    'output_onnx/mnist-custom_4.onnx',  # Output ONNX file
    input_names=['input'],  # Names for the input
    output_names=['output'],  # Names for the output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False  # Optional: Verbose logging
)

# Plot for training process
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
