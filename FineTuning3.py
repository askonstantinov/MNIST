import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d, Span, Label
import numpy as np
import math
import random
import onnx
from onnx2torch import convert


# В данном коде извлечение модели из onnx для ее дообучения здесь производится посредством onnx2torch
# ВАЖНОЕ - при сохранении в onnx теряется информация о наличии слоя Dropout, т.е. дообучение будет без него

# Обеспечение повторяемости результатов (фиксация seed)
allseed = 0
random.seed(allseed)
np.random.seed(allseed)
torch.manual_seed(allseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(allseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Извлечение обученной модели из файла onnx
onnx_model_path = 'output_onnx/FineTuned2.onnx'
onnx_model = onnx.load(onnx_model_path)
torch_model = convert(onnx_model)
model = torch_model
model.to(device)  # Перенос модели на вычислитель (при наличии - на GPU, иначе - на CPU)
print('model=', model)  # Визуальная проверка


def evaluate_model(model, test_loader, device):
    """
    Оценивает модель на тестовой выборке и выводит точность.

    :param model: Обученная модель
    :param test_loader: DataLoader для тестовой выборки
    :param device: Устройство (CPU или GPU)
    :return: Точность модели на тестовой выборке
    """
    model.eval()  # Переводим модель в режим инференса

    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем градиенты для экономии памяти
        for images, labels in test_loader:
            images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
            labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

            outputs = model(images)  # Получаем предсказания модели
            _, predicted = torch.max(outputs.data, 1)  # Находим предсказанные классы
            total += labels.size(0)  # Обновляем общее количество оцененных примеров
            correct += (predicted == labels).sum().item()  # Подсчитываем количество правильных предсказаний

    accuracy = (correct / total) * 100  # Вычисляем точность
    return accuracy


# Ввод значений гиперпараметров (результат серии экспериментов)
num_epochs = 1
batch_size = 256
learning_rate = 5e-07

# Определение путей для данных MNIST
DATA_PATH = 'mnist-data-path'
MODEL_STORE_PATH = 'model-store-path'

# Параметры подготовки данных MNIST
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Широко известная для задач классификации функция потерь на основе кросс-энтропии
criterion = nn.CrossEntropyLoss()

# Оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-05)

# Обучение
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    # model.train()  # Режим обучения - влияет на слои Batch Normalization - здесь не ставлю намеренно

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
        labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        # Вывод промежуточных результатов в процессе обучения после батча
        if batch_size >= total_step and (i + 1) == total_step:
            print(f'Train Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                  f'SUPER Batch = Total steps [{total_step}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
        elif (i + 1) % batch_size == 0:
            print(f'Train Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                  f'Batch [{int((i + 1) / batch_size)}/{math.ceil(total_step / batch_size)}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
        elif (i + 1) == total_step:
            print(f'Train Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                  f'RESIDUAL Batch [{(int((i + 1) / batch_size)) + 1}/{math.ceil(total_step / batch_size)}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')

# Перевод в режим inference (влияет на слои Dropout и Batch Normalization)
# Вывод точности на тестовой выборке после всего обучения
final_test_accuracy = evaluate_model(model, test_loader, device)
print(f'Final Test Accuracy on the 10000 test images: {final_test_accuracy} %')

# Сохранение обученной модели в формат onnx
torch_input = torch.randn(1, 1, 28, 28, device=device)  # Генерируем случайные данные в нашей размерности
torch.onnx.export(
    model,  # Собственно модель
    (torch_input,),  # Инициализация графа вычислений случайными данными в нашей размерности
    'output_onnx/FineTuned3.onnx',  # Расположение и наименование итогового файла onnx
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Поддержка батчей различных размеров
    verbose=False  # Логгирование
)

# Отрисовка процесса обучения с графиками потерь (loss_list) и точности (acc_list)
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch FineTuned OptunaNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list, legend_label="Train Loss", line_color="blue")
p.line(np.arange(len(acc_list)), np.array(acc_list) * 100, y_range_name='Accuracy', legend_label="Train Accuracy", line_color="red")

# Вертикальные линии для разграничения эпох
for j in range(1, (num_epochs + 1)):
    z = j * total_step
    p.add_layout(Span(location=z, dimension='height', line_color='green', line_width=1))
    label = Label(x=z, y=1, text=f"{j}", text_align='right')  # Подписи
    p.add_layout(label)

# Вывод графика на экран (html откроется в браузере)
show(p)
