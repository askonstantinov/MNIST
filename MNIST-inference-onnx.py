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
import onnx
import onnxruntime as ort


batch_size = 100

# Specific for MNIST integrated into PyTorch
DATA_PATH = '/home/konstantinov/PycharmProjects/MNIST/mnist-data-path'
MODEL_STORE_PATH = '/home/konstantinov/PycharmProjects/MNIST/model-store-path'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# Data loader
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        ### Объявим пустой tupple для наполнения в цикле
        out_t = ()

        ### Зададим параметры инициализации onnx модели и откроем сессию onnxruntime с подгрузкой обученной модели onnx
        input = torch.randn(1, 1, 28, 28)
        ort_sess = ort.InferenceSession('/home/konstantinov/PycharmProjects/MNIST/mnist-custom1.onnx')

        ### Автоматизируем процесс присваивания ключей для словаря в цикле
        input_name = ort_sess.get_inputs()[0].name

        for i in range(0, len(images)):
            print('Проверочное значение, должна быть цифра =', int(labels[i]))

            # НИЖЕ - ПРИВЕДЕНИЕ КО ВХОДУ onnxruntime
            k = images[i,:,:]
            z = np.array(k)
            j = np.expand_dims(z, axis=0)

            # НИЖЕ - СТАРТ ЗАМЕРА ВРЕМЕНИ
            start = datetime.datetime.now()

            # НИЖЕ - ИНФЕРЕНС модели onnx силами onnxruntime
            outputs = ort_sess.run(None, {input_name: j})

            # НИЖЕ - КОНЕЦ ЗАМЕРА ВРЕМЕНИ
            finish = datetime.datetime.now()

            outputs_array = outputs[0][0]
            print('Результат инференса: получена цифра =', np.argmax(outputs_array))
            print('Затраты времени на отработку инференса (например, 0:00:00.000118 это 118 мкс): ' + str(finish - start))
            print('--------------------')
            print('Порядковый номер проверенной картинки в текущей пачке (всего 100 картинок) =', i+1)
            print('--------------------')

            outputs = torch.tensor(outputs)[0]
            out_t = torch.cat((torch.tensor(out_t), outputs),0)

        _, predicted = torch.max(out_t.data, 1)
        total += labels.size(0)
        print('########################################')
        print('Номер проверенной пачки тестовых изображений (всего 100 пачек) =', int(total/100))
        print('########################################')
        correct += (predicted == labels).sum().item()

    print('================================================================================')
    print('================================================================================')
    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
