import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d, Span, Label, Legend, LegendItem, Panel
from bokeh.models.layouts import Column, Row
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx2torch import convert
import onnxruntime as ort
import math
import os


# Извлечение модели из onnx для ее дообучения посредством onnx2torch

# Просмотр обученных моделей (графов) https://netron.app/

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Обеспечим повторяемость запусков - заблокируем состояние генератора случайных чисел
# Установка seeds для CPU и GPU
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load onnx
onnx_model_path = 'output_onnx/mnist_seed0_batch_size32_learning_rate5e-05_epoch7.onnx'
onnx_model = onnx.load(onnx_model_path)

# Extract parameters from onnx into pytorch
torch_model = convert(onnx_model)
model = torch_model
model.to(device)  # Перенос модели на устройство GPU
#print('model=', model)

# Hyperparameters for training
num_epochs = 9
batch_size = 32
learning_rate = 1e-03

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

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-04)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на устройство GPU
        labels = labels.to(device)  # Перенос меток на устройство GPU

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
        '''
        if batch_size >= total_step and (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.6f}, Train Accuracy: {:.4f} %'
                .format(epoch + 1, num_epochs, i + 1, total_step,
                        total_step, loss.item(), (correct / total) * 100))
        elif (i + 1) % batch_size == 0:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.6f}, Train Accuracy: {:.4f} %'
                .format(epoch + 1, num_epochs, i + 1, total_step, int((i + 1) / batch_size),
                        math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
        elif (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.6f}, Train Accuracy: {:.4f} %'
                .format(epoch + 1, num_epochs, i + 1, total_step, (int((i + 1) / batch_size)) + 1,
                        math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
        '''

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

        print(f"Epoch {epoch+1} - Test Accuracy of the model on the 10000 test images: {((correct / total) * 100):.4f} %")

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

    print(f"Test Accuracy of the model on the 10000 test images: {((correct / total) * 100):.4f} %")

# Save trained model into onnx
torch_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(
    model,  # PyTorch model
    (torch_input,),  # Input data
    'output_onnx/mnist_seed0_batch_size32_learning_rate5e-05_epoch7_2.onnx',  # Output ONNX file
    input_names=['input'],  # Names for the input
    output_names=['output'],  # Names for the output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False  # Optional: Verbose logging
)

# Save trained model into .pt
#torch.save(model.state_dict(),'output_pt/mnist-custom_piecewise_2.pt')
'''
# Plot for training process
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')

# График потерь (loss_list) и точности обучения (acc_list)
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list, legend_label="Train Loss", line_color="blue")
p.line(np.arange(len(acc_list)), np.array(acc_list) * 100, y_range_name='Accuracy', legend_label="Train Accuracy", line_color="red")

# Добавляем вертикальные линии для обозначения эпох
for i in range(1, (num_epochs + 1)):
    x = i * total_step
    p.add_layout(Span(location=x, dimension='height', line_color='green', line_width=1))

    # Добавляем подписи к линиям
    label = Label(x=x, y=1, text=f"+{i}", text_align='right')
    p.add_layout(label)

# Показываем график
show(p)
'''
os.system('python MNIST-etl-onnx-2.py')