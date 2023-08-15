# Обучение модели классификации комментариев


## Описание проекта

Интернет-магазин запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Требуется инструмент, который будет искать токсичные комментарии и отправлять их на модерацию.


## Навыки и инструменты

- **python**
- **pandas**
- **matplotlib**
- **numpy**
- from **pymystem3** import Mystem
- from nltk.stem.wordnet import **WordNetLemmatizer**
- from nltk.corpus import stopwords as **nltk_stopwords**
- from nltk.corpus import **wordnet**

- **scikit-learn**
  - from sklearn.feature_extraction.text import **CountVectorizer**
  - from sklearn.feature_extraction.text import **TfidfVectorizer**
  - from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score,
    cross_validate
)
  - from sklearn.pipeline import Pipeline, make_pipeline
  - from sklearn.preprocessing import StandardScaler
  - from sklearn.linear_model import LogisticRegression
  - from sklearn.tree import DecisionTreeClassifier
  - from sklearn.ensemble import RandomForestClassifier
  - from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)

## Вывод

Необходимо было построить модель со значением метрики качества *F1* не меньше 0.75. 

Для достижения поставленных целей были выполнены следующие Шаги:

- ШАГ 1. 

В данном пункте:
- изучили датасет. Строк: 159292. Столбцов: 2. столбцы соответсвуют своему типу данных. тексту требуется дополнительная обработка.
- явные дупликаты отсуствуют
- наблюдается дисбаланс классов. Ответов с 0 составляет около 90%
- была выполнена лемматизация и очистка от лишних символов

Для борьбы с дисбалансом принято решение рассмотреть способ class_weight.
Для данных способов была выполнена разбивка выборок на тренироввочную и тестовую. Разбивки в обоих случаях были в соотношении 80:20


- ШАГ 2.

Рассмотрели 3 модели:

- Решающее дерево
- Случайный лес
- Логистическая регрессия

Лучше всего себя показала модель Логистическая регрессия - LogisticRegression с уменьшенным кол-вом выборки. На ней и выполнялось теститорование модели.

Гиперпараметры такой модели: {'model_lr__C': 8, 'model_lr__class_weight': 'balanced', 'model_lr__max_iter': 100, 'model_lr__penalty': 'l2', 'model_lr__solver': 'liblinear'}

Выполнив проверку модели на тестовой выборке получил:

- F1 мера наилучшей модели на тестовой выборке: 0.774025974025974
- Полнота наилучшей модели на тестовой выборке: 0.8285449490268767
- Точность наилучшей модели на тестовой выборке: 0.7262388302193339
- AUC-ROC наилучшей модели на тестовой выборке: 0.8966112349075408

Так как F1 не меньше 0.75, то модель нам подходит. 