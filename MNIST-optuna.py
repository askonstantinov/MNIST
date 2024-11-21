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

'''
################################################################################################
# НИЖЕ ИЗЛОЖЕНА МОЯ КРАТКАЯ ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ OPTUNA С ПОЯСНЕНИЯМИ (НА ОСНОВЕ PYTORCH)
################################################################################################

# Суть Optuna в том, чтобы применить smart подход (вместо грубых эвристик) и автоматизировать процесс поиска
# наилучшей комбинации гиперпараметров (ГП). Поэтому важно представлять полный набор ГП. Типы гиперпараметров:

# 1) ГП модели: включают параметры, определяющие архитектуру модели, например, количество скрытых слоев и нейронов;
    # для Optuna следует задавать модель явно (речь о 'скелете')

# 2) ГП оптимизации: к ним относятся параметры, управляющие процессом оптимизации, например, скорость обучения;
    # для Optuna можно использовать подгруженную модель, например, из onnx

# б/н) ГП регуляризации: имеют отношение, например, к коэффициенту dropout, а также настройкам L1 / L2 регуляризации.
    # Важное - L1 регуляризация (L1 penalty, Lasso) в torch не представлена
    # Важное - L2 регуляризация (L2 penalty, Ridge) в torch задается в настройках оптимизатора как weight_decay,
        # поэтому целесообразно отнести к ГП оптимизации
    # Важное - dropout задается отдельным слоем, поэтому целесообразно отнести к ГП модели.

################################################ 1) Перечень ГП модели:

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
# 'ReLU'  # Опробован
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
# 'LeakyReLU'  # Используется
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

################################################ 2) Перечень ГП оптимизации:

# Выбор оптимизатора градиентного спуска. Для каждого разный набор ГП. Список оптимизаторов torch:
# adadelta
# adagrad
# adam  # Используется
# adamw  # Опробован
# sparse_adam
# adamax
# asgd
# sgd
# radam
# rprop
# rmsprop  # Опробован
# nadam  # Опробован
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
# 'MSELoss'  # Опробован
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
# 'BatchNorm2d'  # Используется
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
'''

# Ниже представлено применение Optuna для автоматического подбора ГП модели и обучение нейросети с лучшими ГП модели


# Целевая функция Optuna
def objective(trial, number_epochs_optuna, criterion_optuna):
    #print('########## Optuna Trial =', trial.number + 1)
    
    # Фиксация необходимых ГП оптимизации (2), включая общие ГП
    num_epochs_optuna = number_epochs_optuna
    learning_rate_optuna = 1e-3
    batch_size_optuna = 32

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

    # ВАЖНОЕ - можно реализовать подбор количества слоев и их расположения, но для этого нужны вычислительные мощности
    # Код с подобным примером представлен в другом скрипте - MNIST-optuna-adv.py

    # Определение конкретных ГП модели (1) с диапазонами для подбора в ходе отработки Optuna
    layer1_conv2d_filter = trial.suggest_int('layer1_conv2d_filter', 128, 256, step=32)
    layer1_conv2d_kernel = trial.suggest_int('layer1_conv2d_kernel', 5, 7, step=2)
    layer1_leakyrelu = trial.suggest_float('layer1_leakyrelu', 1e-03, 2e-01, log=True)

    layer2_conv2d_filter = trial.suggest_int('layer2_conv2d_filter', 128, 256, step=32)
    layer2_conv2d_kernel = trial.suggest_int('layer2_conv2d_kernel', 5, 7, step=2)
    layer2_leakyrelu = trial.suggest_float('layer2_leakyrelu', 1e-03, 2e-01, log=True)

    layer3_conv2d_filter = trial.suggest_int('layer3_conv2d_filter', 128, 256, step=32)
    layer3_conv2d_kernel = trial.suggest_int('layer3_conv2d_kernel', 5, 7, step=2)
    layer3_leakyrelu = trial.suggest_float('layer3_leakyrelu', 1e-03, 2e-01, log=True)

    layer4_conv2d_filter = trial.suggest_int('layer4_conv2d_filter', 128, 256, step=32)
    layer4_conv2d_kernel = trial.suggest_int('layer4_conv2d_kernel', 5, 7, step=2)
    layer4_leakyrelu = trial.suggest_float('layer4_leakyrelu', 1e-03, 2e-01, log=True)

    # Размерность входа полносвязного слоя, стоящего после сверточных, будет зависеть от параметров maxpooling
    layer5_fc1_neurons = layer4_conv2d_filter * int(
        train_loader.dataset.dataset.data.size(1) / (2*2)) * int(train_loader.dataset.dataset.data.size(1) / (2*2))
    layer6_fc2_neurons = trial.suggest_int('layer6_fc2_neurons', 512, 1024, step=128)

    # Формирование 'скелета' нейросети для Optuna в явном виде
    class PreOptunaNet(nn.Module):
        def __init__(self):
            super(PreOptunaNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, layer1_conv2d_filter, kernel_size=layer1_conv2d_kernel, stride=1,
                          padding=int(layer1_conv2d_kernel / 2)),
                nn.LeakyReLU(negative_slope=layer1_leakyrelu),
                nn.BatchNorm2d(layer1_conv2d_filter))
            self.layer2 = nn.Sequential(
                nn.Conv2d(layer1_conv2d_filter, layer2_conv2d_filter, kernel_size=layer2_conv2d_kernel, stride=1,
                          padding=int(layer2_conv2d_kernel / 2)),
                nn.LeakyReLU(negative_slope=layer2_leakyrelu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(layer2_conv2d_filter))
            self.layer3 = nn.Sequential(
                nn.Conv2d(layer2_conv2d_filter, layer3_conv2d_filter, kernel_size=layer3_conv2d_kernel, stride=1,
                          padding=int(layer3_conv2d_kernel / 2)),
                nn.LeakyReLU(negative_slope=layer3_leakyrelu),
                nn.BatchNorm2d(layer3_conv2d_filter))
            self.layer4 = nn.Sequential(
                nn.Conv2d(layer3_conv2d_filter, layer4_conv2d_filter, kernel_size=layer4_conv2d_kernel, stride=1,
                          padding=int(layer4_conv2d_kernel / 2)),
                nn.LeakyReLU(negative_slope=layer4_leakyrelu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(layer4_conv2d_filter))
            self.fc1 = nn.Linear(layer5_fc1_neurons, layer6_fc2_neurons)
            self.fc1act = nn.LeakyReLU()
            self.drop_out = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(layer6_fc2_neurons, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc1act(out)
            out = self.drop_out(out)
            out = self.fc2(out)
            #out = torch.softmax(out, dim=1)  # Вычисление вероятностей с помощью Softmax - для loss MSE
            return out

    model = PreOptunaNet()
    print('Optuna model =', model)  # Визуальная проверка

    model.to(device)  # Перенос модели на вычислитель (при наличии - на GPU, иначе - на CPU)

    # Оптимизатор Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_optuna)

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
            #loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())  # для loss MSE
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
number_epochs_optuna = 10
# Loss
criterion = nn.CrossEntropyLoss()  # Loss для отработки Optuna и для финального полного обучения с лучшими ГП модели (1)
#criterion = nn.MSELoss()  # для работы MSE надо добавить слой softmax в конец (в forward) и добавить one_hot

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=50),
                            pruner=optuna.pruners.HyperbandPruner(),
                            direction='maximize')
study.optimize(lambda trial: objective(trial, number_epochs_optuna, criterion),
               n_trials=1001)  # надо задавать >100 trials в силу статистической природы подходов, реализуемых в Optuna

# Вывод результатов
print(f'Лучшая точность: {study.best_value}')
print(f'Лучшие параметры: {study.best_params}')
print(f'Количество pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}')

################################################################################################
# Ниже представлено полное (без прунов) (валидационный датасет включен в обучающий датасет)
# обучение с наилучшими (определенными выше) параметрами
print('#################### ФИНАЛЬНОЕ ОБУЧЕНИЕ без прунов (валидационный датасет включен в обучающий датасет)')
# Фиксация необходимых ГП оптимизации (2), включая общие ГП
number_epochs_final = 30
learning_rate_final = 1e-3
batch_size_final = 32

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

layer1_conv2d_filter = best_params['layer1_conv2d_filter']
layer1_conv2d_kernel = best_params['layer1_conv2d_kernel']
layer1_leakyrelu = best_params['layer1_leakyrelu']

layer2_conv2d_filter = best_params['layer2_conv2d_filter']
layer2_conv2d_kernel = best_params['layer2_conv2d_kernel']
layer2_leakyrelu = best_params['layer2_leakyrelu']

layer3_conv2d_filter = best_params['layer3_conv2d_filter']
layer3_conv2d_kernel = best_params['layer3_conv2d_kernel']
layer3_leakyrelu = best_params['layer3_leakyrelu']

layer4_conv2d_filter = best_params['layer4_conv2d_filter']
layer4_conv2d_kernel = best_params['layer4_conv2d_kernel']
layer4_leakyrelu = best_params['layer4_leakyrelu']

layer5_fc1_neurons = layer4_conv2d_filter * int(
    train_loader.dataset.data.size(1) / (2*2)) * int(train_loader.dataset.data.size(1) / (2*2))

layer6_fc2_neurons = best_params['layer6_fc2_neurons']


# Задаем модель нейросети в явном виде для финального обучения
class OptunaNet(nn.Module):
    def __init__(self):
        super(OptunaNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, layer1_conv2d_filter, kernel_size=layer1_conv2d_kernel, stride=1,
                      padding=int(layer1_conv2d_kernel / 2)),
            nn.LeakyReLU(negative_slope=layer1_leakyrelu),
            nn.BatchNorm2d(layer1_conv2d_filter))
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_conv2d_filter, layer2_conv2d_filter, kernel_size=layer2_conv2d_kernel, stride=1,
                      padding=int(layer2_conv2d_kernel / 2)),
            nn.LeakyReLU(negative_slope=layer2_leakyrelu),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(layer2_conv2d_filter))
        self.layer3 = nn.Sequential(
            nn.Conv2d(layer2_conv2d_filter, layer3_conv2d_filter, kernel_size=layer3_conv2d_kernel, stride=1,
                      padding=int(layer3_conv2d_kernel / 2)),
            nn.LeakyReLU(negative_slope=layer3_leakyrelu),
            nn.BatchNorm2d(layer3_conv2d_filter))
        self.layer4 = nn.Sequential(
            nn.Conv2d(layer3_conv2d_filter, layer4_conv2d_filter, kernel_size=layer4_conv2d_kernel, stride=1,
                      padding=int(layer4_conv2d_kernel / 2)),
            nn.LeakyReLU(negative_slope=layer4_leakyrelu),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(layer4_conv2d_filter))
        self.fc1 = nn.Linear(layer5_fc1_neurons, layer6_fc2_neurons)
        self.fc1act = nn.LeakyReLU()
        self.drop_out = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(layer6_fc2_neurons, 10)

    def forward(self, xx):
        outout = self.layer1(xx)
        outout = self.layer2(outout)
        outout = self.layer3(outout)
        outout = self.layer4(outout)
        outout = outout.reshape(outout.size(0), -1)
        outout = self.fc1(outout)
        outout = self.fc1act(outout)
        outout = self.drop_out(outout)
        outout = self.fc2(outout)
        # outout = torch.softmax(outout, dim=1)  # Вычисление вероятностей с помощью Softmax - для loss MSE
        return outout


model = OptunaNet()
print('model =', model)  # Визуальная проверка

model = model.to(device)  # Перенос модели на вычислитель (при наличии - на GPU, иначе - на CPU)

# Оптимизатор Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_final)

# Обучение
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(number_epochs_final):
    model.train()  # Режим обучения - влияет на слои Dropout и Batch Normalization

    for ii, (images, labels) in enumerate(train_loader):

        images = images.to(device)  # Перенос данных на вычислитель (при наличии - на GPU, иначе - на CPU)
        labels = labels.to(device)  # Перенос меток на вычислитель (при наличии - на GPU, иначе - на CPU)

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())  # для loss MSE
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
        if batch_size_final >= total_step and (ii + 1) == total_step:
            print(f'Train Epoch [{epoch + 1}/{number_epochs_final}], Step [{ii + 1}/{total_step}], '
                  f'SUPER Batch = Total steps [{total_step}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
        elif (ii + 1) % batch_size_final == 0:
            print(f'Train Epoch [{epoch + 1}/{number_epochs_final}], Step [{ii + 1}/{total_step}], '
                  f'Batch [{int((ii + 1) / batch_size_final)}/{math.ceil(total_step / batch_size_final)}], '
                  f'Loss: {loss.item():.6f}, Train Accuracy: {((correct / total) * 100):.2f} %')
        elif (ii + 1) == total_step:
            print(f'Train Epoch [{epoch + 1}/{number_epochs_final}], Step [{ii + 1}/{total_step}], '
                  f'RESIDUAL Batch [{(int((ii + 1) / batch_size_final)) + 1}/{math.ceil(total_step / batch_size_final)}], '
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
    'output_onnx/OptunaNet_0.onnx',  # Расположение и наименование итогового файла onnx
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Поддержка батчей различных размеров
    verbose=False  # Логгирование
)

# Сохранение обученной модели в формат .pt (это формат Python)
torch.save(model.state_dict(),'output_pt/OptunaNet_0.pt')

# Отрисовка процесса обучения с графиками потерь (loss_list) и точности (acc_list)
p = figure(y_axis_label='Loss', width=1700, y_range=(0, 1), title='PyTorch OptunaNet_0 results')
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
