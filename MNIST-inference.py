import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import datetime


batch_size = 100

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

# Neural network (two CNN layers with ReLU and maxpool2d, dropout, two fc layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()

# Test the model
model.eval()

# Load pt
model.load_state_dict(torch.load('output_pt/mnist-custom_1.pt'))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        for i in range(0, len(images)):
            print('Проверочное значение, должна быть цифра =', int(labels[i]))

            # СТАРТ ЗАМЕРА ВРЕМЕНИ
            start = datetime.datetime.now()

            # ИНФЕРЕНС модели pt
            outputs = model(images)

            # КОНЕЦ ЗАМЕРА ВРЕМЕНИ
            finish = datetime.datetime.now()

            # Вывод результата инференса в каждой из 100 тест-пачек
            print_outputs = outputs[i]
            print_max_outputs = np.argmax(print_outputs)
            print('Результат инференса: получена цифра =', print_max_outputs.item())
            print('Затраты времени на отработку инференса (например, 0:00:00.000123 это 123 мкс): ' + str(finish - start))
            print('--------------------')
            print('Порядковый номер проверенной картинки в текущей пачке (всего 100 картинок) =', i+1)
            print('--------------------')

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print('########################################')
        print('Номер проверенной пачки тестовых изображений (всего 100 пачек) =', int(total/100))
        print('########################################')
        correct += (predicted == labels).sum().item()

    print('================================================================================')
    print('================================================================================')
    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
