import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import math


# Перед первым запуском - убедитесь, что создана виртуальная среда (например, conda).
# Установите в среду все необходимое командой
# pip install -r requirements.txt
# Не забудьте активировать подготовленную среду!

# Просмотр обученных моделей (графов) https://netron.app/

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Hyperparameters for training
num_epochs = 5
num_classes = 10
batch_size = 64
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.84, 0.9995), eps=4e-09, weight_decay=1e-05)

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
    'output_onnx/mnist-custom_piecewise_1.onnx',  # Output ONNX file
    input_names=['input'],  # Names for the input
    output_names=['output'],  # Names for the output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False  # Optional: Verbose logging
)

# Save trained model into .pt
#torch.save(model.state_dict(),'output_pt/mnist-custom_piecewise_1.pt')

# Plot for training process
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
