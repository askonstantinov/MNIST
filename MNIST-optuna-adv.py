import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d, Span, Label
import numpy as np
import torch
import optuna
import math
from torch.utils.data import random_split
import random


# В настоящем скрипте реализован подбор средствами Optuna количества слоев и их расположения,
# но следует учесть, что для запуска кода потребуются значительные вычислительные мощности

# Перед первым запуском - проверить корректность активированной виртуальной среды.
# При необходимости - создать виртуальную среду (conda), установить все необходимое командой
# pip install -r requirements.txt
# После подготовки виртуальной среды - активировать ее.

# Просмотреть сохраненные обученные модели (pt или onnx) можно тут
# https://netron.app/

# Обеспечение повторяемости результатов (фиксация seed)
# Для наилучшего результата следует экспериментировать с разными значениями seed (эвристика)
allseed = 10
random.seed(allseed)
np.random.seed(allseed)
torch.manual_seed(allseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(allseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'Используемое устройство: {device}')

# МОЯ КРАТКАЯ ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ OPTUNA С ПОЯСНЕНИЯМИ (НА ОСНОВЕ PYTORCH) - ПРИВЕДЕНА В СКРИПТЕ
# MNIST-optuna.py

# Целевая функция Optuna
def objective(trial, number_epochs_optuna, criterion_optuna):
    # print('########## Optuna Trial =', trial.number + 1)

    # Фиксация необходимых ГП
    num_epochs_optuna = number_epochs_optuna
    learning_rate_optuna = 1e-3
    batch_size_optuna = 32

    adam_betas1 = 0.8447
    adam_betas2 = 0.9995
    adam_eps = 4.578e-09
    adam_weight_decay = 1.3e-05

    # Формирование массивов данных MNIST из базы данных PyTorch
    DATA_PATH = 'mnist-data-path'
    MODEL_STORE_PATH = 'model-store-path'
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Выделение 60000 образцов для обучения и валидации, а 10000 - для теста (датасет MNIST содержит 70000 образцов)
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    # Разделение на обучающий и валидационный датасеты (в соотношении обучение-валидация-тест примерно как 70%-15%-15%)
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_optuna, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_optuna, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_optuna, shuffle=False)

    # Формирование 'скелета' нейросети для Optuna в явном виде
    class PreOptunaNetAdv(nn.Module):
        def __init__(self, conv_layers, fc_layers):
            super(PreOptunaNetAdv, self).__init__()
            self.conv_layers = nn.Sequential(*conv_layers)
            self.fc_layers = nn.Sequential(*fc_layers)
            self.fc_act = nn.LeakyReLU()
            self.drop_out = nn.Dropout(p=0.5)
            self.fc_end = nn.Linear(fc_layers[-1].out_features, 10)

        def forward(self, x):
            out = self.conv_layers(x)
            out = out.reshape(out.size(0), -1)
            out = self.fc_layers(out)
            out = self.fc_act(out)
            out = self.drop_out(out)
            out = self.fc_end(out)
            return out

    # Определение слоев
    n_layers_fc = trial.suggest_int("n_layers_fc", 1, 4)  # Определение числа полносвязных слоев
    conv_layers = []
    fc_layers = []
    # Создание слоев
    conv_layers.append(nn.Conv2d(1, 224, kernel_size=5, stride=1, padding=2))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.012))
    conv_layers.append(nn.BatchNorm2d(224))
    conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.121))
    conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    conv_layers.append(nn.BatchNorm2d(224))
    conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.033))
    conv_layers.append(nn.BatchNorm2d(224))
    conv_layers.append(nn.Conv2d(224, 160, kernel_size=5, stride=1, padding=2))
    conv_layers.append(nn.LeakyReLU(negative_slope=0.095))
    conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    conv_layers.append(nn.BatchNorm2d(160))

    for i in range(n_layers_fc):
        in_features = 7840 if i == 0 else fc_layers[-1].out_features
        out_features = trial.suggest_categorical(f"fc_out_{i}", [128, 256, 512, 1024])
        fc_layers.append(nn.Linear(in_features, out_features))

    model = PreOptunaNetAdv(conv_layers, fc_layers)
    print('Optuna model =', model)  # Визуальная проверка

    model.to(device)  # Перенос модели на вычислитель (при наличии - на GPU, иначе - на CPU)

    # Оптимизатор Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_optuna, betas=(adam_betas1, adam_betas2), eps=adam_eps, weight_decay=adam_weight_decay)

    # Loss
    criterion = criterion_optuna

    # Обучение модели для выбранной конфигурации гиперпараметров Optuna
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    val_acc_list = []
    for epoch in range(num_epochs_optuna):
        model.train()  # Режим обучения - влияет на слои Dropout и Batch Normalization

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
            labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

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
            # Вывод промежуточных результатов в процессе обучения после батча
            if batch_size_optuna >= total_step and (i + 1) == total_step:
                print(f'Optuna Train Epoch [{epoch + 1}/{num_epochs_optuna}], Step [{i + 1}/{total_step}], '
                      f'SUPER Batch = Total steps [{total_step}], '
                      f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
            elif (i + 1) % batch_size_optuna == 0:
                print(f'Optuna Train Epoch [{epoch + 1}/{num_epochs_optuna}], Step [{i + 1}/{total_step}], '
                      f'Batch [{int((i + 1) / batch_size_optuna)}/{math.ceil(total_step / batch_size_optuna)}], '
                      f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
            elif (i + 1) == total_step:
                print(f'Optuna Train Epoch [{epoch + 1}/{num_epochs_optuna}], Step [{i + 1}/{total_step}], '
                      f'RESIDUAL Batch [{(int((i + 1) / batch_size_optuna)) + 1}/{math.ceil(total_step / batch_size_optuna)}], '
                      f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
            '''

        # Кросс-валидация после прохождения одной эпохи
        model.eval()  # Перевод в режим inference (влияет на слои Dropout и Batch Normalization)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:

                images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
                labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

                # Вывод точности на валидационной выборке
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = correct / total
            val_acc_list.append(val_acc)
            print(f'Optuna Trial: [{trial.number + 1}]')
            print(f'Processed Epoch: [{epoch + 1}/{num_epochs_optuna}]')
            print(f'Cross-Validation Accuracy: [{(val_acc * 100):.2f} %]')

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()  # ВАЖНОЕ - прун выполняется на основе оценки валидационной точности

    # Тестирование модели
    model.eval()  # Перевод в режим inference (влияет на слои Dropout и Batch Normalization)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:

            images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
            labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

            # Вывод точности на тестовой выборке после всего обучения данного OPTUNA TRIAL
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = (correct / total) * 100
    print(f'Optuna Test Accuracy on the 10000 test images: {test_accuracy:.2f} %')

    # Возврат точности как метрики для Optuna
    return test_accuracy

# Ввод значений параметров и запуск Optuna
number_epochs_optuna = 2
# Loss
criterion = nn.CrossEntropyLoss()  # Loss для отработки Optuna и для финального полного обучения с лучшими ГП

search_space = {
    'n_layers_fc': [1, 4],  # Число полносвязных слоев
    'fc_out_0': [128, 256, 512, 1024, 2048],  # Размерность выхода для 1 полносвязного слоя
    'fc_out_1': [256, 512, 1024],  # Размерность выхода для 2 полносвязного слоя
    'fc_out_2': [128, 256, 512],  # Размерность выхода для 3 полносвязного слоя
    'fc_out_3': [128, 256, 512, 1024],  # Размерность выхода для 4 полносвязного слоя
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space),
                            pruner=optuna.pruners.MedianPruner(),
                            direction='maximize')
study.optimize(lambda trial: objective(trial, number_epochs_optuna, criterion),
               n_trials=3)

# Вывод результатов
print(f'Лучшая точность: {study.best_value}')
print(f'Лучшие параметры: {study.best_params}')
print(f'Количество pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}')

################################################################################################
# Ниже представлено полное (без прунов) (валидационный датасет включен в обучающий датасет)
# обучение с наилучшими (определенными выше) параметрами
print('#################### ФИНАЛЬНОЕ ОБУЧЕНИЕ без прунов (валидационный датасет включен в обучающий датасет)')
# Фиксация необходимых ГП
number_epochs_final = 2
learning_rate_final = 1e-3
batch_size_final = 32

adam_betas1_final = 0.8447
adam_betas2_final = 0.9995
adam_eps_final = 4.578e-09
adam_weight_decay_final = 1.3e-05

# Определение путей для данных MNIST
DATA_PATH = 'mnist-data-path'
MODEL_STORE_PATH = 'model-store-path'

# Параметры подготовки данных MNIST
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=False)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_final, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_final, shuffle=False)

# Ввод определенных Optuna лучших параметров
best_params = study.best_params
n_layers_fc = best_params['n_layers_fc']


# Задаем модель нейросети в явном виде для финального обучения
class OptunaNetAdv(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super(OptunaNetAdv, self).__init__()
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


# Определение слоев
conv_layers = []
fc_layers = []
# Создание слоев
conv_layers.append(nn.Conv2d(1, 224, kernel_size=5, stride=1, padding=2))
conv_layers.append(nn.LeakyReLU(negative_slope=0.012))
conv_layers.append(nn.BatchNorm2d(224))
conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
conv_layers.append(nn.LeakyReLU(negative_slope=0.121))
conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
conv_layers.append(nn.BatchNorm2d(224))
conv_layers.append(nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3))
conv_layers.append(nn.LeakyReLU(negative_slope=0.033))
conv_layers.append(nn.BatchNorm2d(224))
conv_layers.append(nn.Conv2d(224, 160, kernel_size=5, stride=1, padding=2))
conv_layers.append(nn.LeakyReLU(negative_slope=0.095))
conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
conv_layers.append(nn.BatchNorm2d(160))

for ii in range(n_layers_fc):
    in_features = 7840 if ii == 0 else fc_layers[-1].out_features
    out_features = best_params['fc_out' + str(f'_{ii}')]
    fc_layers.append(nn.Linear(in_features, out_features))

# Создадим итоговую модель
model = OptunaNetAdv(conv_layers, fc_layers)
print('Optuna model =', model)  # Визуальная проверка

model.to(device)  # Перенос модели на вычислитель (при наличии - на GPU, иначе - на CPU)

# Оптимизатор Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_final, betas=(adam_betas1_final, adam_betas2_final), eps=adam_eps_final, weight_decay=adam_weight_decay_final)

# Обучение
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(number_epochs_final):
    model.train()  # Режим обучения - влияет на слои Dropout и Batch Normalization

    for iii, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
        labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

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

        # Вывод промежуточных результатов в процессе обучения после батча
        if batch_size_final >= total_step and (iii + 1) == total_step:
            print(f'Train Epoch [{epoch + 1}/{number_epochs_final}], Step [{iii + 1}/{total_step}], '
                  f'SUPER Batch = Total steps [{total_step}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
        elif (iii + 1) % batch_size_final == 0:
            print(f'Train Epoch [{epoch + 1}/{number_epochs_final}], Step [{iii + 1}/{total_step}], '
                  f'Batch [{int((iii + 1) / batch_size_final)}/{math.ceil(total_step / batch_size_final)}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
        elif (iii + 1) == total_step:
            print(f'Train Epoch [{epoch + 1}/{number_epochs_final}], Step [{iii + 1}/{total_step}], '
                  f'RESIDUAL Batch [{(int((iii + 1) / batch_size_final)) + 1}/{math.ceil(total_step / batch_size_final)}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')

# Перевод в режим inference (влияет на слои Dropout и Batch Normalization)
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
        labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

        # Вывод точности на тестовой выборке после всего обучения
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy on the 10000 test images: {(correct / total) * 100:.2f} %')

# Сохранение обученной модели в формат onnx
torch_input = torch.randn(1, 1, 28, 28, device=device)  # Генерируем случайные данные в нашей размерности
torch.onnx.export(
    model,  # Собственно модель
    (torch_input,),  # Инициализация графа вычислений случайными данными в нашей размерности
    'output_onnx/OptunaNetAdv_0.onnx',  # Расположение и наименование итогового файла onnx
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Поддержка батчей различных размеров
    verbose=False  # Логгирование
)

# Сохранение обученной модели в формат .pt (это формат Python)
torch.save(model.state_dict(),'output_pt/OptunaNetAdv_0.pt')

# Отрисовка процесса обучения с графиками потерь (loss_list) и точности (acc_list)
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch OptunaNetAdv_0 results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list, legend_label='Train Loss', line_color='blue')
p.line(np.arange(len(acc_list)), np.array(acc_list) * 100,
       y_range_name='Accuracy', legend_label='Train Accuracy', line_color='red')

# Вертикальные линии для разграничения эпох
for j in range(1, (number_epochs_final + 1)):
    z = j * total_step
    p.add_layout(Span(location=z, dimension='height', line_color='green', line_width=1))
    label = Label(x=z, y=1, text=f'{j}', text_align='right')  # Подписи
    p.add_layout(label)

# Вывод графика на экран (html откроется в браузере)
show(p)
