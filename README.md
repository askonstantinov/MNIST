Цель ветки optuna - показать, как можно применить популярный инструмент автоматического подбора гиперпараметров OPTUNA в задаче MNIST.
Для демонстрации возможностей я подготовил скрипты MNIST-optuna.py и MNIST-optuna-adv.py
===
1) Благодаря MNIST-optuna.py , перебирая различные значения seed, средствами Optuna и была выявлена конфигурация ГП модели (1) для архитектуры, с которой ручной подбор ГП оптимизации (2) (правда, с некоторыми эвристиками в рамках системного подхода) позволил получить в итоге решение, которое показывает на тестовом датасете MNIST точность 99.72 % и здесь важно отметить, что Я НЕ ПРИМЕНЯЛ АУГМЕНТАЦИЮ (это можно сделать дополнительно).

Более подробно, как я получил итоговое решение, описано в commit from Ubuntu 19-10-2024:
Архитектура нейросети предварительно подобрана настроенной Optuna (настройки определены экспериментально) в течение нескольких суток непрерывной работы Core i5 13400 + MSI Nvidia RTX3060 8GB ("ГП модели" - количество слоев, количество фильтров и нейронов, вид и положение функций активации, настройки и положение слоев maxpooling, слои batchnorm и dropout).
Настройки обучения ("ГП оптимизации" - количество скриптов-итераций дообучения, количество эпох обучения каждого скрипта-итерации дообучения, mini-batch size каждого скрипта-итерации дообучения, learning rate каждого скрипта-итерации дообучения, параметры оптимизатора каждого скрипта-итерации дообучения, значение seed для каждого скрипта-итерации дообучения) - определены экспериментально, отработаны и проверены.
Запуск цепочки скриптов OptunaNet.py - FineTuning1.py - FineTuning2.py - FineTuning3.py производится запуском первого скрипта OptunaNet.py
В итоге будет получен FineTuned3.onnx - эта обученная модель показывает на тестовых 10000 изображениях MNIST высокую точность 99.72 %
ВАЖНО - данный результат получен без аугментации и дополнительных манипуляций с обучающими 60000 избражениями MNIST.
Если посмотреть опубликованный перечень лучших решений для MNIST (последнее - от 2012 года), то можно заметить, что среди решений без дополнительного препроцессинга наилучший имеет точность всего лишь 99.65 %
https://yann.lecun.com/exdb/mnist/

2) Суть Optuna в том, чтобы применить smart подход (вместо грубых эвристик) и автоматизировать процесс поиска наилучшей комбинации гиперпараметров (ГП).
Поэтому важно представлять полный набор ГП.
Типы гиперпараметров:
ГП модели: включают параметры, определяющие архитектуру модели, например, количество скрытых слоев и нейронов;
для Optuna следует задавать модель явно (речь о 'скелете').
ГП оптимизации: к ним относятся параметры, управляющие процессом оптимизации, например, скорость обучения;
для Optuna можно использовать подгруженную модель, например, из onnx.
ГП регуляризации: имеют отношение, например, к коэффициенту dropout, а также настройкам L1 / L2 регуляризации.
Важное - L1 регуляризация (L1 penalty, Lasso) в torch не представлена
Важное - L2 регуляризация (L2 penalty, Ridge) в torch задается в настройках оптимизатора как weight_decay, поэтому целесообразно отнести к ГП оптимизации
Важное - dropout задается отдельным слоем, поэтому целесообразно отнести к ГП модели
3) В коде MNIST-optuna-adv.py продемонстрировано, как применить Оптуну к подбору количества полносвязных слоев с соответствующим количеством искусственных нейронов. После выполнения всех Trials запускается обучение с наилучшей конфигурацией.
Такой подход создает высокую вычислительную нагрузку на "железо" (у меня вычисления производились на cuda - MSI RTX 3060 8GB), поэтому пришлось сильно ограничить пространство поиска, применить "перебор по сетке из возможных значений" вместо поиска силами TPE самплера, а также настроить median pruner вместо более продвинутых прунеров.
По аналогии можно настроить подбор также для количества и расположения сверточных слоев и/или любых других слоев и ГП вообще. Ограничения - мощность "железа".
Из обнаруженных ограничений - dropout слой не всегда корректно отрабатывает в связке со слоем batchnorm.
4) Общие комментарии:
Осталась проблема с lock для seed (несмотря на мои попытки - новые запуски по-разному инициализируют стартовую конфигурацию Оптуны - для скрипта adv это выражается в разном начальном количестве подбираемых слоев).
Для улучшения результатов в задаче MNIST можно также добавить dropout после каждого слоя.
