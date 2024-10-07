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
import math
from torch.utils.data import random_split


# Перед первым запуском - убедитесь, что создана виртуальная среда (например, conda).
# Установите в среду все необходимое командой
# pip install -r requirements.txt
# Не забудьте активировать подготовленную среду!

# Просмотр обученных моделей (графов) https://netron.app/

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Обеспечим повторяемость запусков - заблокируем состояние генератора случайных чисел
# Установка seeds для CPU и GPU

# При batch_size = 128 и learning_rate = 1e-03 получено для разных seed:
# seedX = 0  # 10 epochs get 98.25 % on test
# seedX = 1  # 10 epochs get 99.09 % on test
# seedX = 2  # 10 epochs get 99.01 % on test
# seedX = 3  # 10 epochs get 98.89 % on test
# seedX = 4  # 10 epochs get 98.81 % on test
# seedX = 5  # 10 epochs get 98.99 % on test
# seedX = 6  # 10 epochs get 98.95 % on test
# seedX = 7  # 10 epochs get 98.96 % on test
# seedX = 8  # 10 epochs get 98.97 % on test
# seedX = 9  # 10 epochs get 99.19 % on test
# seedX = 10  # 10 epochs get 99.10 % on test

# При batch_size = 128 и learning_rate = 5e-04 получил:
# seedX = 0  # 10 epochs get 98.87 % on test

# При batch_size = 128 и learning_rate = 1e-04 получил:
# seedX = 0  # 10 epochs get 98.98 % on test
# seedX = 0  # 9 epochs get 99.18 % on test  # нужно смотреть по кросс-валидации - и вовремя делать останова!

# При batch_size = 64 и learning_rate = 1e-03 получил:
# seedX = 0  # 10 epochs get 99.13 % on test
# seedX = 1  # 10 epochs get 99.19 % on test
# seedX = 10  # 10 epochs get 99.06 % on test - это отличается от полученного ранее 99.37 %, т.к. здесь я "отщипнул" от датасета 10000 на валидацию

seedX = 0
torch.manual_seed(seedX)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seedX)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters for training
num_epochs = 40
num_classes = 10

# Заведем в таблицу значений batch и lr те, что составляют best practice:
#batch_size = 32  # a - нет смысла учить подольше, нужно сразу прорабатывать piecewise lr через onnx-файлы
batch_size = 64  # b
#batch_size = 128  # c - есть смысл учить подольше

#learning_rate = 1e-03  # 1
#learning_rate = 2e-03  # 2
#learning_rate = 2.5e-03  # 3
#learning_rate = 5e-03  # 4
#learning_rate = 7.5e-03  # 5

#learning_rate = 1e-04  # 6
#learning_rate = 2e-04  # 7
#learning_rate = 2.5e-04  # 8
#learning_rate = 5e-04  # 9
#learning_rate = 7.5e-04  # 10

#learning_rate = 1e-05  # 11
#learning_rate = 2e-05  # 12
#learning_rate = 2.5e-05  # 13
#learning_rate = 5e-05  # 14
learning_rate = 7.5e-05  # 15

'''
Проведем grid search для соотношений batch-lr на протяжении 10 эпох. 
Выберем и сохраним в onnx для последующего piecewise lr эксперимента 
наилучшие модели с ранней останова по критерию точности cross-val или test >=99.20% 
на протяжении 10 эпох (для батча размером 32) и 40 эпох (для больших батчей).

Cross-val results and test results for seed=0:
 a1 на 10 эпох: best_epoch_8=99.08% test_epoch_10=99.30%  # a1 на 8 эпох save
a2 на 10 эпох: best_epoch_8=98.97% test_epoch_10=99.08%
 a3 на 10 эпох: best_epoch_10=98.99% test_epoch_10=99.25%  # a3 на 10 эпох save
a4 на 10 эпох: best_epoch_10=99.05% test_epoch_10=99.07%
a5 на 10 эпох: best_epoch_4=98.55% test_epoch_10=98.40%
 a6 на 10 эпох: best_epoch_10=99.31% test_epoch_10=99.33%  # a6 на 10 эпох save
a7 на 10 эпох: best_epoch_7=99.19% test_epoch_10=98.93%
 a8 на 10 эпох: best_epoch_9=99.23% test_epoch_10=98.76%  # a8 на 9 эпох save
a9 на 10 эпох: best_epoch_6=99.17% test_epoch_10=99.16%
 a10 на 10 эпох: best_epoch_8=99.34% test_epoch_10=99.23%  # a10 на 8 эпох save
 a11 на 10 эпох: best_epoch_8=99.19% test_epoch_10=99.29%  # a11 на 8 эпох save
 a12 на 10 эпох: best_epoch_10=99.28% test_epoch_10=99.27%  # a12 на 10 эпох save
 a13 на 10 эпох: best_epoch_7=99.25% test_epoch_10=99.22%  # a13 на 7 эпох save
 a14 на 10 эпох: best_epoch_7=99.34% test_epoch_10=98.69%  # a14 на 7 эпох save
 a15 на 10 эпох: best_epoch_6=99.25% test_epoch_10=99.15%  # a15 на 6 эпох save

b1 на 10 эпох: best_epoch_8=99.05% test_epoch_10=99.13%
b2 на 10 эпох: best_epoch_8=98.97% test_epoch_10=99.08%
b3 на 10 эпох: best_epoch_8=99.12% test_epoch_10=98.78%
b4 на 10 эпох: best_epoch_9=99.01% test_epoch_10=98.70%
b5 на 10 эпох: best_epoch_8=98.88% test_epoch_10=98.61%
b6 на 10 эпох: best_epoch_9=99.13% test_epoch_10=99.05%
b7 на 10 эпох: best_epoch_6=99.18% test_epoch_10=98.31%
 b8 на 10 эпох: best_epoch_10=99.23% test_epoch_10=99.13%  # b8 на 40 эпох: best_epoch_
b9 на 10 эпох: best_epoch_9=99.05% test_epoch_10=98.89%
b10 на 10 эпох: best_epoch_7=99.09% test_epoch_10=98.86%
 b11 на 10 эпох: best_epoch_9=99.25% test_epoch_10=99.30%  # b11 на 40 эпох: best_epoch_32=99.46%
 b12 на 10 эпох: best_epoch_8=99.26% test_epoch_10=99.01%  # b12 на 40 эпох: best_epoch_35=99.49%
 b13 на 10 эпох: best_epoch_9=99.34% test_epoch_10=99.21%  # b13 на 40 эпох: best_epoch_37=99.41%
b14 на 10 эпох: best_epoch_10=99.12% test_epoch_10=99.07%
 b15 на 10 эпох: best_epoch_6=99.23% test_epoch_10=99.20%  # b15 на 40 эпох: best_epoch_30=99.47%

c1 на 10 эпох: best_epoch_5=99.10% test_epoch_10=98.25%
c2 на 10 эпох: best_epoch_8=98.84% test_epoch_10=98.78%
c3 на 10 эпох: best_epoch_9=98.96% test_epoch_10=98.91%
c4 на 10 эпох: best_epoch_7=98.68% test_epoch_10=98.57%
c5 на 10 эпох: best_epoch_9=98.73% test_epoch_10=98.32%
 c6 на 10 эпох: best_epoch_9=99.24% test_epoch_10=98.98%  # c6 на 40 эпох:
 c7 на 10 эпох: best_epoch_5=99.22% test_epoch_10=99.16%  # c7 на 40 эпох:
 c8 на 10 эпох: best_epoch_7=99.20% test_epoch_10=99.19%  # c8 на 40 эпох:
 c9 на 10 эпох: best_epoch_6=99.20% test_epoch_10=98.87%  # c9 на 40 эпох:
c10 на 10 эпох: best_epoch_7=99.04% test_epoch_10=98.37%
 c11 на 10 эпох: best_epoch_7=99.19% test_epoch_10=99.20%  # c11 на 40 эпох: best_epoch_30=99.42%
 c12 на 10 эпох: best_epoch_7=99.25% test_epoch_10=99.19%  # c12 на 40 эпох: best_epoch_31=99.48%
 c13 на 10 эпох: best_epoch_4=99.23% test_epoch_10=99.26%  # c13 на 40 эпох: best_epoch_39=99.51% save
 c14 на 10 эпох: best_epoch_10=99.30% test_epoch_10=99.23%  # c14 на 40 эпох: best_epoch_29=99.51% save
 c15 на 10 эпох: best_epoch_10=99.28% test_epoch_10=99.23%  # c15 на 40 эпох: best_epoch_27=99.47%
'''
# Есть смысл опробовать 40 эпох для батча размером 128 при условии, что 10 эпох - вполне достаточно для батча размером 32

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

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Neural network (two CNN layers with ReLU and maxpool2d, dropout, two fc layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 224, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.012921133981887153),
            nn.BatchNorm2d(224))
        self.layer2 = nn.Sequential(
            nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(negative_slope=0.12110839449567463),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(224))
        self.layer3 = nn.Sequential(
            nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(negative_slope=0.03359161678276241),
            nn.BatchNorm2d(224))
        self.layer4 = nn.Sequential(
            nn.Conv2d(224, 160, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.09512672917154825),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(160))
        self.fc1 = nn.Linear(160 * 7 * 7, 1024)
        self.fc1act = nn.LeakyReLU()
        self.drop_out = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return x


model = ConvNet()
model.to(device)  # Перенос модели на устройство GPU
print('model=', model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-04)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
val_acc_list = []
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
                'Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, num_epochs, i + 1, total_step,
                        total_step, loss.item(), (correct / total) * 100))
        elif (i + 1) % batch_size == 0:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, num_epochs, i + 1, total_step, int((i + 1) / batch_size),
                        math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
        elif (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, num_epochs, i + 1, total_step, (int((i + 1) / batch_size)) + 1,
                        math.ceil(total_step / batch_size), loss.item(), (correct / total) * 100))
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
        print(f"########################### Train Epoch [{epoch+1}/{num_epochs}], Cross-Validation Accuracy: {(val_acc*100):.2f} %")
        if (val_acc*100) > 99.49:
            # Save trained model into onnx
            torch_input = torch.randn(1, 1, 28, 28, device=device)
            torch.onnx.export(
                model,  # PyTorch model
                (torch_input,),  # Input data
                f'output_onnx/mnist_seed{seedX}_batch_size{batch_size}_learning_rate{learning_rate}_epoch{epoch+1}.onnx',  # Output ONNX file
                input_names=['input'],  # Names for the input
                output_names=['output'],  # Names for the output
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                verbose=False  # Optional: Verbose logging
            )

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
'''
# Save trained model into onnx
torch_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(
    model,  # PyTorch model
    (torch_input,),  # Input data
    'output_onnx/mnist-custom_piecewise_1.onnx',  # Output ONNX file
    input_names=['input'],  # Names for the input
    output_names=['output'],  # Names for the output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False  # Optional: Verbose logging
)
'''
# Save trained model into .pt
#torch.save(model.state_dict(),'output_pt/mnist-custom_piecewise_1.pt')

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

# График кросс-валидации (val_acc_list)
val_indices = [(i + 1) * total_step for i in range(num_epochs)]
p.scatter(val_indices, np.array(val_acc_list) * 100, size=8,
         color="black", y_range_name='Accuracy', legend_label="Validation Accuracy")
p.line(val_indices, np.array(val_acc_list) * 100,
        line_color="black", y_range_name='Accuracy')

# Добавляем вертикальные линии для обозначения эпох
for i in range(1, (num_epochs + 1)):
    x = i * total_step
    p.add_layout(Span(location=x, dimension='height', line_color='green', line_width=1))

    # Добавляем подписи к линиям
    label = Label(x=x, y=1, text=f"#{i}", text_align='right')
    p.add_layout(label)

# Показываем график
show(p)
