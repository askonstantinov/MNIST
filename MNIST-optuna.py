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
#device = torch.device("cpu")
print(f"Используемое устройство: {device}")

################################################################################################

# Суть Optuna в том, чтобы применить smart подход (вместо грубых эвристик) и автоматизировать процесс поиска
# наилучшей комбинации гиперпараметров (ГП). Поэтому важно представлять полный набор ГП. Типы гиперпараметров:

# 1) ГП модели: включают параметры, определяющие архитектуру модели, например, количество скрытых слоев и нейронов;
    # для Optuna следует задавать модель явно

# 2) ГП оптимизации: к ним относятся параметры, управляющие процессом оптимизации, например, скорость обучения;
    # для Optuna можно использовать подгруженную модель, например, из onnx

# б/н) ГП регуляризации: имеют отношение, например, к коэффициенту dropout, а также настройкам L1 / L2 регуляризации.
    # Важное - L1 регуляризация (L1 penalty, Lasso) в torch не представлена
    # Важное - L2 регуляризация (L2 penalty, Ridge) в torch задается в настройках оптимизатора как weight_decay,
        # поэтому целесообразно отнести к ГП оптимизации
    # Важное - dropout задается отдельным слоем, поэтому целесообразно отнести к ГП модели.

################################################ 1) Перечень:

# Тип слоя, его положение, количество нейронов,
# Количество слоев,
# Функция активации и ее положение.

# Согласно библиотеке torch доступны следующие слои:
# 'Bilinear'
# 'Identity'
# 'LazyLinear'
# 'Linear'  # Используется
# 'Conv1d'
# 'Conv2d'  # Используется
# 'Conv3d'
# 'ConvTranspose1d'
# 'ConvTranspose2d'
# 'ConvTranspose3d'
# 'LazyConv1d'
# 'LazyConv2d'
# 'LazyConv3d'
# 'LazyConvTranspose1d'
# 'LazyConvTranspose2d'
# 'LazyConvTranspose3d'
# 'RNNBase'
# 'RNN'
# 'LSTM'
# 'GRU'
# 'RNNCellBase'
# 'RNNCell'
# 'LSTMCell'
# 'GRUCell'
# 'Transformer'
# 'TransformerEncoder'
# 'TransformerDecoder'
# 'TransformerEncoderLayer'
# 'TransformerDecoderLayer'
#
# Для используемого слоя 'Conv2d' доступны следующие настройки:
# in_channels (int): Number of channels in the input image
# out_channels (int): Number of channels produced by the convolution
# kernel_size (int or tuple): Size of the convolving kernel
# stride (int or tuple, optional): Stride of the convolution. Default: 1
# padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0
# padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
# dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
# groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
# bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
#
# Кроме того, для 'Conv2d' доступны вспомогательные слои пулинга. Список torch:
# 'MaxPool1d'
# 'MaxPool2d'  # Используется
# 'MaxPool3d'
# 'MaxUnpool1d'
# 'MaxUnpool2d'
# 'MaxUnpool3d'
# 'AvgPool1d'
# 'AvgPool2d'
# 'AvgPool3d'
# 'FractionalMaxPool2d'
# 'FractionalMaxPool3d'
# 'LPPool1d'
# 'LPPool2d'
# 'LPPool3d'
# 'AdaptiveMaxPool1d'
# 'AdaptiveMaxPool2d'
# 'AdaptiveMaxPool3d'
# 'AdaptiveAvgPool1d'
# 'AdaptiveAvgPool2d'
# 'AdaptiveAvgPool3d'
#
# Для используемого 'MaxPool2d' доступны настройки:
# kernel_size: the size of the window to take a max over
# stride: the stride of the window. Default value is :attr:`kernel_size`
# padding: Implicit negative infinity padding to be added on both sides
# dilation: a parameter that controls the stride of elements in the window
# return_indices: if ``True``, will return the max indices along with the outputs
# ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
#
# Список функций активации torch:
# 'Threshold'
# 'ReLU'  # Используется
# 'RReLU'
# 'Hardtanh'
# 'ReLU6'
# 'Sigmoid'
# 'Hardsigmoid'
# 'Tanh'
# 'SiLU'
# 'Mish'
# 'Hardswish'
# 'ELU'
# 'CELU'
# 'SELU'
# 'GLU'
# 'GELU'
# 'Hardshrink'
# 'LeakyReLU'
# 'LogSigmoid'
# 'Softplus'
# 'Softshrink'
# 'MultiheadAttention'
# 'PReLU'
# 'Softsign'
# 'Tanhshrink'
# 'Softmin'
# 'Softmax'
# 'Softmax2d'
# 'LogSoftmax'

################################################ 2) Перечень:

# Выбор оптимизатора градиентного спуска. Для каждого разный набор ГП. Список оптимизаторов torch:
# adadelta
# adagrad
# adam  # Используется
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

# Для используемого оптимизатора adam список ГП:
# params (iterable): iterable of parameters to optimize or dicts defining parameter groups
# lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR is not yet supported for all our
    # implementations. Please use a float LR if you are not also specifying fused=True or capturable=True.
# betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient
    # and its square (default: (0.9, 0.999))
# eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
# weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
# amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm from the paper
    # `On the Convergence of Adam and Beyond`_ (default: False)

# К общим ГП можно отнести:
# количество эпох (полных проходов по всем обучающим данным),
# размер пачки (batch_size) и батч-нормализация,
# loss функция.

# Список loss функций torch:
# 'L1Loss'
# 'NLLLoss'
# 'NLLLoss2d'
# 'PoissonNLLLoss'
# 'GaussianNLLLoss'
# 'KLDivLoss'
# 'MSELoss'  # Опробую
# 'BCELoss'
# 'BCEWithLogitsLoss'
# 'HingeEmbeddingLoss'
# 'MultiLabelMarginLoss'
# 'SmoothL1Loss'
# 'HuberLoss'
# 'SoftMarginLoss'
# 'CrossEntropyLoss'  # Используется
# 'MultiLabelSoftMarginLoss'
# 'CosineEmbeddingLoss'
# 'MarginRankingLoss'
# 'MultiMarginLoss'
# 'TripletMarginLoss'
# 'TripletMarginWithDistanceLoss'
# 'CTCLoss'

# Список вариантов батч-нормализации:
# 'BatchNorm1d'
# 'LazyBatchNorm1d'
# 'BatchNorm2d'
# 'LazyBatchNorm2d'
# 'BatchNorm3d'
# 'LazyBatchNorm3d'
# 'SyncBatchNorm'

# Для 'BatchNorm2d' список настроек:
# num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`
# eps: a value added to the denominator for numerical stability. Default: 1e-5
# momentum: the value used for the running_mean and running_var computation. Can be set to ``None`` for cumulative
    # moving average (i.e. simple average). Default: 0.1
# affine: a boolean value that when set to ``True``, this module has learnable affine parameters. Default: ``True``
# track_running_stats: a boolean value that when set to ``True``, this module tracks the running mean and variance,
    # and when set to ``False``, this module does not track such statistics, and initializes statistics buffers :attr:
    # running_mean` and :attr:`running_var` as ``None``. When these buffers are ``None``, this module always
    # uses batch statistics in both training and eval modes. Default: ``True``

################################################ б/н) Перечень:

# Регуляризация с использованием слоя dropout - список torch:
# 'Dropout'  # Используется
# 'Dropout1d'
# 'Dropout2d'
# 'Dropout3d'
# 'AlphaDropout'
# 'FeatureAlphaDropout'

# Для 'Dropout' список настроек:
# p: probability of an element to be zeroed. Default: 0.5
# inplace: If set to ``True``, will do this operation in-place. Default: ``False``

################################################################################################

# Просто загнать все возможные ГП и все значения - плохая идея.
# Следует примерно представлять, какие ГП важны, а также разумные диапазоны значений.

# Помимо ГП, можно поработать с  dataset (Data Augmentation, нормализация etc),
# поменять настройки Optuna (sampler, pruner, number of trials etc),
# либо изменить постановку задачи.

# Предлагается следующая последовательность применения Optuna:
# - сначала отработать по ГП модели,
# - затем для наилучших моделей отработать по ГП оптимизации.

# В любом случае эффективное применение Optuna связано с огромным объемом вычислений, почти все из которых
# дадут результаты, подлежащие отсеиванию. Поэтому не рекомендуется делать всё на одном вычислителе.

################################################################################################

# Целевая функция Optuna
def objective(trial, path_to_onnx_model_optuna, number_epochs_optuna, criterion_optuna):
    # Fixed hyperparameters needed for training
    num_epochs_optuna = number_epochs_optuna
    learning_rate_optuna = 1e-3
    batch_size_optuna = 32

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

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_optuna, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_optuna, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_optuna, shuffle=False)

    # Range of hyperparameters to choose from Optuna (1)
    layer1_conv2d_filter = trial.suggest_int('layer1_conv2d_filter', 32, 128)
    layer1_conv2d_kernel = trial.suggest_int('layer1_conv2d_kernel', 3, 7, step=2)
    layer1_leakyrelu = trial.suggest_float('layer1_leakyrelu', 1e-03, 1e-01, log=True)

    layer2_conv2d_filter = trial.suggest_int('layer2_conv2d_filter', 32, 128)
    layer2_conv2d_kernel = trial.suggest_int('layer2_conv2d_kernel', 3, 7, step=2)
    layer2_leakyrelu = trial.suggest_float('layer2_leakyrelu', 1e-03, 1e-01, log=True)

    layer3_fc1_neurons = layer2_conv2d_filter * int(train_loader.dataset.dataset.data.size(1) / (2*2)) * int(train_loader.dataset.dataset.data.size(1) / (2*2))

    layer4_fc2_neurons = trial.suggest_int('layer4_fc2_neurons', 100, 10000)

    # Задаем модель нейросети для Optuna в явном виде
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, layer1_conv2d_filter, kernel_size=layer1_conv2d_kernel, stride=1, padding=int(layer1_conv2d_kernel / 2)),
                nn.BatchNorm2d(layer1_conv2d_filter),
                nn.LeakyReLU(negative_slope=layer1_leakyrelu),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(layer1_conv2d_filter, layer2_conv2d_filter, kernel_size=layer2_conv2d_kernel, stride=1, padding=int(layer2_conv2d_kernel / 2)),
                nn.BatchNorm2d(layer2_conv2d_filter),
                nn.LeakyReLU(negative_slope=layer2_leakyrelu),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc1 = nn.Linear(layer3_fc1_neurons, layer4_fc2_neurons)
            self.drop_out = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(layer4_fc2_neurons, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop_out(out)
            out = self.fc2(out)
            #out = torch.softmax(out, dim=1)  # Вычисление вероятностей с помощью Softmax
            return out

    model = ConvNet()
    print(model)

    model.to(device)  # Перенос модели на устройство GPU

    # Определение оптимизатора
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_optuna)

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
            #loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
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
onnxpath = 'output_onnx/mnist-custom_1.onnx'
number_epochs_optuna = 14
# Loss
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()  # для работы MSE нужно добавить слой softmax в конец (в forward) и добавить в цикл one_hot

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=30),
                            pruner=optuna.pruners.HyperbandPruner(),
                            direction='maximize')
study.optimize(lambda trial: objective(trial, onnxpath, number_epochs_optuna, criterion),
               n_trials=101)  # желательно задавать >100 trials

# Вывод результатов
print(f"Лучшая точность: {study.best_value}")
print(f"Лучшие параметры: {study.best_params}")
print(f"Количество обрезанных (pruned) trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

################################################################################################
# Обучение модели с лучшими параметрами
# Ввод прочих параметров
number_epochs_final = 14
learning_rate_final = 1e-3
batch_size_final = 32

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

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_final, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_final, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_final, shuffle=False)

# Ввод определенных Optuna лучших параметров
best_params = study.best_params
layer1_conv2d_filter = best_params['layer1_conv2d_filter']
layer1_conv2d_kernel = best_params['layer1_conv2d_kernel']
layer1_leakyrelu = best_params['layer1_leakyrelu']
layer2_conv2d_filter = best_params['layer2_conv2d_filter']
layer2_conv2d_kernel = best_params['layer2_conv2d_kernel']
layer2_leakyrelu = best_params['layer2_leakyrelu']
layer3_fc1_neurons = layer2_conv2d_filter * int(train_loader.dataset.dataset.data.size(1) / (2*2)) * int(train_loader.dataset.dataset.data.size(1) / (2*2))
layer4_fc2_neurons = best_params['layer4_fc2_neurons']

# Задаем модель нейросети в явном виде для финального обучения
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, layer1_conv2d_filter, kernel_size=layer1_conv2d_kernel, stride=1, padding=int(layer1_conv2d_kernel / 2)),
            nn.BatchNorm2d(layer1_conv2d_filter),
            nn.LeakyReLU(negative_slope=layer1_leakyrelu),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_conv2d_filter, layer2_conv2d_filter, kernel_size=layer2_conv2d_kernel, stride=1, padding=int(layer2_conv2d_kernel / 2)),
            nn.BatchNorm2d(layer2_conv2d_filter),
            nn.LeakyReLU(negative_slope=layer2_leakyrelu),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(layer3_fc1_neurons, layer4_fc2_neurons)
        self.drop_out = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(layer4_fc2_neurons, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        #out = torch.softmax(out, dim=1)  # Вычисление вероятностей с помощью Softmax
        return out


model = ConvNet()
print(model)

model = model.to(device)  # Перенос модели на устройство GPU

# Определение оптимизатора
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_final)

# Обучение модели
total_step = len(train_loader)
loss_list = []
acc_list = []
val_acc_list = []
for epoch in range(number_epochs_final):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на устройство
        labels = labels.to(device)  # Перенос меток на устройство

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
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

        if batch_size_final >= total_step and (i + 1) == total_step:
            print(
                'Train Epoch [{}/{}], Step [{}/{}], SUPER Batch = Total steps [{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                .format(epoch + 1, number_epochs_final, i + 1, total_step,
                        total_step, loss.item(), (correct / total) * 100))
        elif (i + 1) % batch_size_final == 0:
            print('Train Epoch [{}/{}], Step [{}/{}], Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
                  .format(epoch + 1, number_epochs_final, i + 1, total_step, int((i + 1) / batch_size_final),
                          math.ceil(total_step / batch_size_final), loss.item(), (correct / total) * 100))
        elif (i + 1) == total_step:
            print('Train Epoch [{}/{}], Step [{}/{}], RESIDUAL Batch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f} %'
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
        print(f"########################### Cross-Validation Accuracy: {(val_acc*100):.2f} %")

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
    'output_onnx/mnist-custom_1.onnx',  # Output ONNX file
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
