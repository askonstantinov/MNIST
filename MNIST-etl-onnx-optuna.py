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


# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Используемое устройство: {device}")

# Суть Optuna в том, чтобы применить smart подход (вместо грубых эвристик)
# и автоматизировать процесс поиска наилучшей комбинации гиперпараметров. Типы гиперпараметров:
# 1) ГП модели: включают параметры, определяющие архитектуру модели, например, количество скрытых слоев;
# 2) ГП оптимизации: к ним относятся параметры, управляющие процессом оптимизации, например, скорость обучения;
# 3) ГП регуляризации: имеют отношение, например, к коэффициенту dropout, а также настройкам L1 / L2 регуляризации.

# 1) Перечень:


# 2) Перечень:

# Выбор оптимизатора градиентного спуска. Для каждого разный набор ГП. Список оптимизаторов torch:
# adadelta
# adagrad
# adam
# adamw
# sparse_adam
# adamax
# asgd
# sgd
# radam
# rprop
# rmsprop
# nadam
# lbfgs

# Для популярного оптимизатора градиентного спуска ADAM из библиотеки torch список ГП:
# lr: Union[float, Tensor] = 1e-3
# betas: Tuple[float, float] = (0.9, 0.999)
# eps: float = 1e-8
# weight_decay: float = 0
# amsgrad: bool = False
# foreach: Optional[bool] = None
# maximize: bool = False
# capturable: bool = False
# differentiable: bool = False
# fused: Optional[bool] = None

# К общим ГП можно отнести:
# количество эпох (полных проходов по всем обучающим данным),
# размер пачки (batch_size),
# loss функция.

# Список loss функций torch:
# 'L1Loss'
# 'NLLLoss'
# 'NLLLoss2d'
# 'PoissonNLLLoss'
# 'GaussianNLLLoss'
# 'KLDivLoss'
# 'MSELoss'
# 'BCELoss'
# 'BCEWithLogitsLoss'
# 'HingeEmbeddingLoss'
# 'MultiLabelMarginLoss'
# 'SmoothL1Loss'
# 'HuberLoss'
# 'SoftMarginLoss'
# 'CrossEntropyLoss'
# 'MultiLabelSoftMarginLoss'
# 'CosineEmbeddingLoss'
# 'MarginRankingLoss'
# 'MultiMarginLoss'
# 'TripletMarginLoss'
# 'TripletMarginWithDistanceLoss'
# 'CTCLoss'

# 3) Перечень:


# Просто загнать все возможные ГП и все значения - плохая идея.
# Следует хоть примерно представлять, какие ГП важны и какие разумные диапазоны значений.
# Всегда следует помнить о том, что, помимо ГП, можно "поработать" еще с  dataset (например, Data Augmentation),
# поменять настройки самой Optuna, либо вообще переработать постановку задачи.
# В любом случае эффективное применение Optuna связано с огромным объемом вычислений, почти все из которых
# дадут результаты, подлежащие отсеиванию. Поэтому не рекомендуется делать всё на одном вычислителе.

# Целевая функция Optuna
def objective(trial, path_to_onnx_model_optuna, number_epochs_optuna, criterion_optuna):
    # Range of hyperparameters to choose from Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    # Нужно учесть, что возможны случаи batch_size > total_step
    # тогда обучение будет вестись как batch_size = total_step
    batch_size = trial.suggest_int('batch_size', 64, 64)
    # Fixed hyperparameters needed for training
    num_epochs = number_epochs_optuna

    # Создание модели
    model = convert(onnx.load(path_to_onnx_model_optuna))  # Загрузка модели из ONNX
    model.to(device)  # Перенос модели на устройство GPU

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
    criterion = criterion_optuna

    # Обучение модели для выбранной конфигурации гиперпараметров Optuna
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
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

            if batch_size >= total_step and (i + 1) == total_step:
                print('Optuna Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step,
                              total_step, loss.item(), (correct / total) * 100))
            elif (i + 1) % batch_size == 0:
                print('Optuna Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, int((i + 1) / batch_size),
                              math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
            elif (i + 1) == total_step:
                print('Optuna Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, (int((i + 1) / batch_size)) + 1,
                              math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))

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

    accuracy = (correct / total) * 100

    # Возврат точности как метрики для Optuna
    return accuracy

# Ввод значений параметров и запуск Optuna
onnxpath = 'output_onnx/mnist-custom_1.onnx'
number_epochs_optuna = 4
# Loss
criterion = nn.CrossEntropyLoss()

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=10),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
                            direction='maximize')
study.optimize(lambda trial: objective(trial, onnxpath, number_epochs_optuna, criterion),
               n_trials=101)  # желательно задавать >100 trials

# Вывод результатов
print(f"Лучшая точность: {study.best_value}")
print(f"Лучшие параметры: {study.best_params}")
print(f"Количество обрезанных (pruned) trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

# Обучение модели с лучшими параметрами
# Ввод определенных Optuna лучших параметров
best_params = study.best_params
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']

# Ввод прочих параметров
numepochs = 4

# Подгрузка исходной модели onnx
onnx_model = onnx.load(onnxpath)
# Формирование массивов данных MNIST
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

# Extract parameters from onnx into pytorch
torch_model = convert(onnx_model)
model = torch_model
model = model.to(device) # Перенос модели на устройство GPU

# Определение оптимизатора
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(numepochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на устройство
        labels = labels.to(device)  # Перенос меток на устройство

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if batch_size >= total_step and (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, numepochs, i + 1, total_step,
                        total_step, loss.item(), (correct / total) * 100))
        elif (i + 1) % batch_size == 0:
            print('Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, numepochs, i + 1, total_step, int((i + 1) / batch_size),
                          math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
        elif (i + 1) == total_step:
            print('Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, numepochs, i + 1, total_step, (int((i + 1) / batch_size)) + 1,
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
    'output_onnx/mnist-custom_3.onnx',  # Output ONNX file
    input_names=['input'],  # Names for the input
    output_names=['output'],  # Names for the output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False  # Optional: Verbose logging
)

# Plot for training process
p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
