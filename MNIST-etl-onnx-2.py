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


# Скрипт на основе
# https://neurohive.io/ru/tutorial/cnn-na-pytorch/
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/conv_net_py_torch.py

# Извлечение модели из onnx для ее дообучения посредством onnx2torch

# Просмотр обученных моделей (графов) https://netron.app/

# Load onnx
#onnx_model_path = '/home/konstantinov/PycharmProjects/MNIST/output_onnx/mnist-custom_1.onnx'
onnx_model_path = '/home/konstantinov/PycharmProjects/MNIST/external_onnx/mnist.onnx'
onnx_model = onnx.load(onnx_model_path)

# Extract parameters from onnx into pytorch
torch_model = convert(onnx_model)
model = torch_model

# Hyperparameters for training
num_epochs = 30
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Specific for MNIST integrated into PyTorch
DATA_PATH = '/home/konstantinov/PycharmProjects/MNIST/mnist-data-path'
MODEL_STORE_PATH = '/home/konstantinov/PycharmProjects/MNIST/model-store-path'

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Test the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save trained model into onnx
torch_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,  # PyTorch model
    (torch_input,),  # Input data
    'output_onnx/mnist-custom_2.onnx',  # Output ONNX file
    input_names=['input'],  # Names for the input
    output_names=['output'],  # Names for the output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False  # Optional: Verbose logging
)

# Save trained model into .pt
# torch.save(model.state_dict(),'output_pt/mnist-custom_2.pt')

# Plot for training process
p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
