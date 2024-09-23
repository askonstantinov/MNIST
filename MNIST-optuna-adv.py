import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import torch
import optuna
import math
from torch.utils.data import random_split


# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Целевая функция Optuna
def objective(trial, number_epochs_optuna, criterion_optuna):
    # Fixed hyperparameters needed for training
    num_epochs_optuna = number_epochs_optuna
    learning_rate_optuna = 0.00012125747732631978
    batch_size_optuna = 128

    adam_betas1 = 0.8447311629853445
    adam_betas2 = 0.9995301881902142
    adam_eps = 4.578951182620495e-09
    adam_weight_decay = 1.3005998870242143e-05

    # Формирование массивов данных MNIST
    # Specific for MNIST integrated into PyTorch
    DATA_PATH = 'mnist-data-path'
    MODEL_STORE_PATH = 'model-store-path'
    # Transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST 70000 images dataset (60000 images for train, and 10000 images for test)
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    # Разделение на обучающий, валидационный и тестовый наборы (в соотношении приблизительно 70%-15%-15%)
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # Loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_optuna, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_optuna, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_optuna, shuffle=False)

    class MyModel(nn.Module):
        def __init__(self, conv_layers, fc_layers):
            super(MyModel, self).__init__()
            self.conv_layers = nn.Sequential(*conv_layers)
            self.fc_layers = nn.Sequential(*fc_layers)
            self.fc_activation = nn.LeakyReLU()
            self.drop_out = nn.Dropout(p=0.5)
            self.fc_end = nn.Linear(fc_layers[-1].out_features, 10)

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc_layers(x)
            x = self.fc_activation(x)
            x = self.drop_out(x)
            x = self.fc_end(x)
            return x


    # Определим слои
    n_layers_fc = trial.suggest_int("n_layers", 1, 2)  # Определение числа полносвязных слоев
    conv_layers = []
    fc_layers = []
    # Создание слоев
    conv_layers.append(nn.Conv2d(1, 224, kernel_size=5, stride=1, padding=2))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.012921133981887153))
    conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.12110839449567463))
    conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.03359161678276241))
    conv_layers.append(nn.Conv2d(224, 160, kernel_size=5, stride=1, padding=2))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.09512672917154825))
    conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    for i in range(n_layers_fc):
        in_features = 7840 if i == 0 else fc_layers[-1].out_features
        out_features = trial.suggest_categorical(f"fc_out_{i}", [128, 256, 512, 1024, 2048])
        fc_layers.append(nn.Linear(in_features, out_features))

    # Создадим итоговую модель
    model = MyModel(conv_layers, fc_layers)
    model.to(device)  # Перенос модели на устройство GPU
    print('model=', model)

    # Определение оптимизатора
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_optuna, betas=(adam_betas1, adam_betas2), eps=adam_eps, weight_decay=adam_weight_decay)

    # Loss
    criterion = criterion_optuna

    # Обучение модели для выбранной конфигурации гиперпараметров Optuna
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    val_acc_list = []
    for epoch in range(num_epochs_optuna):
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

            '''
            if batch_size_optuna >= total_step and (i + 1) == total_step:
                print('Optuna Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Optuna Train Accuracy: {:.2f} %'
                      .format(epoch + 1, num_epochs_optuna, i + 1, total_step,
                              total_step, loss.item(), (correct / total) * 100))
            elif (i + 1) % batch_size_optuna == 0:
               print('Optuna Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Optuna Train Accuracy: {:.2f} %'
                     .format(epoch + 1, num_epochs_optuna, i + 1, total_step, int((i + 1) / batch_size_optuna),
                             math.ceil(total_step / batch_size_optuna), loss.item(), (correct / total) * 100))
            elif (i + 1) == total_step:
                print('Optuna Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Optuna Train Accuracy: {:.2f} %'
                     .format(epoch + 1, num_epochs_optuna, i + 1, total_step, (int((i + 1) / batch_size_optuna)) + 1,
                              math.ceil(total_step / batch_size_optuna), loss.item(), (correct / total) * 100))
            '''

        # Кросс-валидация
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:

                images = images.to(device)  # Перенос данных на устройство GPU
                labels = labels.to(device)  # Перенос данных на устройство GPU

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = correct / total
            val_acc_list.append(val_acc)
            print(f"########################### Optuna Cross-Validation Accuracy: {(val_acc*100):.2f} %")

        trial.report(val_acc, epoch)
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
number_epochs_optuna = 30
# Loss
criterion = nn.CrossEntropyLoss()

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=50),
                            pruner=optuna.pruners.HyperbandPruner(),
                            direction='maximize')
study.optimize(lambda trial: objective(trial, number_epochs_optuna, criterion),
               n_trials=501)  # желательно задавать >100 trials

# Вывод результатов
print(f"Номер лучшей попытки: Trial {study.best_trial.number}")
print(f"Лучшая точность: {study.best_value}")
print(f"Лучшие параметры: {study.best_params}")
print(f"Количество обрезанных (pruned) trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

################################################################################################

# Обучение модели с лучшими параметрами
# Ввод прочих параметров
number_epochs_final = 30
learning_rate_final = 0.00012125747732631978
batch_size_final = 128

adam_betas1 = 0.8447311629853445
adam_betas2 = 0.9995301881902142
adam_eps = 4.578951182620495e-09
adam_weight_decay = 1.3005998870242143e-05

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
# Разделение на обучающий, валидационный и тестовый наборы (в соотношении приблизительно 70%-15%-15%)
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
# Loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_final, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_final, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_final, shuffle=False)

# Ввод определенных Optuna лучших параметров
best_params = study.best_params
n_layers_fc = best_params['n_layers_fc']

class MyModel(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super(MyModel, self).__init__()
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Sequential(*fc_layers)
        self.fc_activation = nn.LeakyReLU()
        self.drop_out = nn.Dropout(p=0.5)
        self.fc_end = nn.Linear(fc_layers[-1].out_features, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.fc_activation(x)
        x = self.drop_out(x)
        x = self.fc_end(x)
        return x


# Определим слои
conv_layers = []
fc_layers = []
# Создание слоев
conv_layers.append(nn.Conv2d(1, 224, kernel_size=5, stride=1, padding=2))
conv_layers.append(nn.LeakyReLU(negative_slope=0.012921133981887153))
conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
conv_layers.append(nn.LeakyReLU(negative_slope=0.12110839449567463))
conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
conv_layers.append(nn.LeakyReLU(negative_slope=0.03359161678276241))
conv_layers.append(nn.Conv2d(224, 160, kernel_size=5, stride=1, padding=2))
conv_layers.append(nn.LeakyReLU(negative_slope=0.09512672917154825))
conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

for i in range(n_layers_fc):
    in_features = 7840 if i == 0 else fc_layers[-1].out_features
    out_features = best_params['fc_out' + str(f'_{i}')]
    fc_layers.append(nn.Linear(in_features, out_features))

# Создадим итоговую модель
model = MyModel(conv_layers, fc_layers)
model.to(device)  # Перенос модели на устройство GPU
print('model=', model)

# Определение оптимизатора
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_final, betas=(adam_betas1, adam_betas2), eps=adam_eps, weight_decay=adam_weight_decay)

# Обучение модели для выбранной конфигурации гиперпараметров Optuna
total_step = len(train_loader)
loss_list = []
acc_list = []
val_acc_list = []
for epoch in range(number_epochs_final):
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

        if batch_size_final >= total_step and (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, number_epochs_final, i + 1, total_step,
                        total_step, loss.item(), (correct / total) * 100))
        elif (i + 1) % batch_size_final == 0:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, number_epochs_final, i + 1, total_step, int((i + 1) / batch_size_final),
                        math.ceil(total_step / batch_size_final), loss.item(), (correct / total) * 100))
        elif (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, number_epochs_final, i + 1, total_step, (int((i + 1) / batch_size_final)) + 1,
                        math.ceil(total_step / batch_size_final), loss.item(), (correct / total) * 100))

    # Кросс-валидация
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)  # Перенос данных на устройство GPU
            labels = labels.to(device)  # Перенос данных на устройство GPU

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_acc_list.append(val_acc)
        print(f"########################### Cross-Validation Accuracy: {(val_acc * 100):.2f} %")

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
