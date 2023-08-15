# Исследование технологического процесса очистки золота


## Описание проекта

Строится модель машинного обучения для промышленной компании, разрабатывающая решения для эффективной работы промышленных предприятий. Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды на основе данных с параметрами добычи и очистки. Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками. 


## Навыки и инструменты

- **python**
- **pandas**
- **matplotlib**
- **numpy**
- **seaborn**
- **re** 
- **lightgbm** 
- **catboost**
- **scikit-learn**
    - from sklearn.model_selection import (
    **GridSearchCV**,
    **train_test_split**,
    **cross_val_score**
) 
    - from sklearn.preprocessing import **StandardScaler**
    - from sklearn.dummy import **DummyRegressor**
    - from sklearn.linear_model import **LinearRegression**
    - from sklearn.metrics import **mean_absolute_error**

## Вывод

По проекту нужно построить модель, которая предскажет температуру стали. Данная модель необходима для металлургического комбината. Комбинат решил оптимизировать производственные расходы за счет уменьшения потребления электроэнергии на этапе обработки металла.

Для достижения поставленной цели были выполнены следующие шаги:

**Шаг 1**

На данном этапе были загружены датафреймы:

- `data_arc_new.csv` — данные об электродах;
- `data_bulk_new.csv` — данные о подаче сыпучих материалов (объём);
- `data_bulk_time_new.csv` *—* данные о подаче сыпучих материалов (время);
- `data_gas_new.csv` — данные о продувке сплава газом;
- `data_temp_new.csv` — результаты измерения температуры;
- `data_wire_new.csv` — данные о проволочных материалах (объём);
- `data_wire_time_new.csv` — данные о проволочных материалах (время).

Во всех файлах столбец `key` содержит номер партии. В файлах может быть несколько строк с одинаковым значением `key`: они соответствуют разным итерациям обработки.

По выведеным данным можно сказать следующее:

1. data_arc_new
- Строк: 14876. Столбцов: 5
- Кол-во столбцов с пропусками: 0. 
- Уникальные дупликаты отсутсвуют
- Кол-во партий: 3214
- Значения в столбцах "Начало нагрева дугой" и "Конец нагрева дугой" необходимо будет перевести в datetime
- Данные представлены за май-сентябрь 2019 года. По представленными датам и времени можно вычислить длительность нагрева дугой для каждого дйствия в партии и кол-во итераций в партии.
- Наблюдаются аномальные значения. Так в реактивной мощности всречаются аномальные значения вплодь до -715.479924. Для реактивной мощности в нашем случае это не подходит.Вероятно это ошибка измерения и партии с значением ниже нуля необходимо будет удалить
- необходимо будет столбцы все переименовать на англ язык.
- необходимо будет рассчитать полную мощность 


2. data_bulk_new
- датасет содержит  3129 строк и 16 столбцами
- В датасете присутсвует большое кол-во пропусков. Вероятно значения некоторых видов сыпучих материалов не добавлялось в партию. И следовательно пустые значения можно будет заполнить 0.
- Дубликаты также отсутсвуют
- Кол-во партий: 3129
- Аномальные значения не наблюдаются.
- столбцы не записаны по правилам хорошего тона. Надо будет их скорректировать.

3. data_gas_new
- датасет содержит  3239 строк и 2 столбца
- Пропуски в датасете отсутсвуют
- Дубликаты также отсутсвуют
- Кол-во партий: 3129
- Аномальные значения не наблюдаются.
- Данные распределены равномерно.
- необходимо будет столбец "Газ 1" переименовать.

4. data_temp_new
- датасет содержит  18092 строк и 3 столбца
- В датасете присутсвует пропуски. Вероятно датчик не фиксировал температуру. Данные партии необходимо будет удалить. Стоит обратить внимание, что пропуски активно начали возникать с 2500 парти
- - Температура плавления стали находится в диапазоне от 1450 до 1535°С. Следовательно все температуры ниже 1450 необходимо отсеч.
- Дубликаты также отсутсвуют
- Кол-во партий: 3216
- Столбцы "Температура" и "Время замера" необходимо будет переименовать
- Столбец "Время замера" измеить тип данных на datetime, а "Температура" можно перевести к целочисленному типу данных

5. data_wire_new
- датасет содержит  3081 строк и 10 столбцов
- В датасете присутсвует большое кол-во пропусков. Вероятно некоторые виды проволочных материалов не добавлялось в партию. Данные можно заполнить 0 
- Дубликаты также отсутсвуют
- Кол-во партий: 3081
- Аномальные значения не наблюдаются.
- столбцы не записаны по правилам хорошего тона. Надо будет их скорректировать.


data_bulk_time_new и data_wire_time_new оказались не информативными и было принято решение от них отказаться.
Замечено, что во всех данных столбцы не записаны по правилам хорошего тона. Надо будет их скорректировать.

**Шаг 2**

 В ходе выполнения данного пункта было выполнено следующие действия:
 - Переведены названия столбцов data_arc_new, data_bulk_new, data_gas_new, data_temp_new, data_wire_new в правильному виду
 - Выполнено заполнение пустых строк в столбцах data_bulk_new,data_temp_new, data_wire_new нулевым знаечнием.
 - В столбцах data_arc_new и data_temp_new значения к нужному типу данных.
 - Выполнил удаление неинформативных строк, которые являются очевидными ошибками.
    - Датасет data_arc_new: устранил парти с строками с отрицательным знаечнием в столбце reactive_power
    - Датасет data_temp_new:В данном датасете удалил партии, где в строках встречается температруа с 0 значением и значение температуры ниже 1450.
 - Выделил признаки для модели:
    - так для начала в датасете data_arc_new сформировал столбцы 
        - full_power - Полная мощность вычисленная по столбцам reactive_power и active_power в рамках одной итерации
        - time_duration - длительность подаваемой мощности в рамках одной итерации
    - Затем  была сформирована сводная таблица data_arc_new_pivot по партиям с столбцами:
         - key	- номер партии
         - total_power	- суммарная полная мощность  в партии 
         - iterations	- кол-во итераций в партии 
         - total_time_duration - суммарная длительность подаваемой мощности в партии
    - Из датасета data_temp_new сформировал сводную таблицу  data_temp_new_pivot с столбцами:
         - key	- номер партии
         - first_temp	- температру ковша в начале процесса
         - last_temp - температру ковша в конце процесса
         Предварительно отсортировал данные по столбцу time_measuring по возрастанию.
 - Выполнили объединение нужных данных в одну таблицу data_arc_new_pivot. Объединение происходило по столбцу key. 
     - Объединенная таблицы содержит следующие столбцы: total_power,	iterations,	total_time_duration,	first_temp,	last_temp,	bulk_1,	bulk_2,	bulk_3,	bulk_4,	bulk_5,	bulk_6,	bulk_7,	bulk_8,	bulk_9,	bulk_10,	bulk_11,	bulk_12, bulk_13,	bulk_14,	bulk_15,	wire_1,	wire_2,	wire_3,	wire_4,	wire_6,	wire_7,	wire_8,	wire_9,	gas_1
     - Удалил столбец wire_5, так как в нем все значения равны 0
     - Кол-во строк и столбцов собранной таблицы 2324 и 30
 - Исслдеовал объединенную таблицу на мультикореллиарность. Выыявлена сильная взаимовязь выще 0.9 между столбцами wire_8 и bulk_9.
     - Удален столбец bulk_9 
     - Также удалены столбцы 'wire_7','bulk_8', так как в них 1-4 зачения больше 0.
     - Кол-во строк и столбцов собранной таблицы после очистки выше 2324 и 26 соответсвенно
 
 - Выделил в данной таблице признаки: total_power,	iterations,	total_time_duration,	first_temp,	bulk_1,	bulk_2,	bulk_3,	bulk_4,	bulk_5,	bulk_6,	bulk_7,	bulk_10,	bulk_11,	bulk_12, bulk_13,	bulk_14,	bulk_15,	wire_1,	wire_2,	wire_3,	wire_4,	wire_6,	wire_8,	wire_9,	gas_1
 - Выделил целевой признак  last_temp.
 - Выполнил разделение собранной нашей выборки на обучающуюся и тестовую. (в соотношения 75:25).
 - Размер признаков: 
     - на обучающей выборке (1743, 25)
     - на тестовй выборке (581, 25)
 - Размер целевого признака:
     - на обучающей выборке (1743,) 
     - на тестовй выборке (581,)
 - Выполнил масштабирование данных с помощью StandardScaler.
 
 
 **Шаг 3**
 
 - Обучены модели: LinearRegression, CatBoostRegressor, LGBMRegressor. Значения перебирались с помощью GridSearch. 
 - На основе  расчетов лучшей оказалсь модель СatBoost с гипермпараметром 
     - best_params: {'depth': 6, 'iterations': 50, 'learning_rate': 0.1, 'random_state': 50623}.
     
**Шаг 4**

   - MAE на тестовой выборке лучшей модели: 6.484210545372362
   - Оценка адекватности модели с помощью контсантной модели. МАЕ константной модели на тестовой выборке: 8.121556301428965
   - MAE на тестовой выборке меньше 6.8. Что означет что условие проекта выполнено. Относительно константной модели знаечение тоже оказалось меньше.
   - На основе анализа важности признаков модели-победителя  можно выделить длительность водаваемой энергии (total_time_duration) и изначальная температура(first_temp)

