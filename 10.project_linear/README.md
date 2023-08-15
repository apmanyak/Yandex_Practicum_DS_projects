# Защита данных клиентов страховой компании


## Описание проекта

Необходимо защитить данные клиентов страховой компании.Необходимо разработать метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию.  Нужно защитить данные, чтобы при преобразовании качество моделей машинного обучения не ухудшилось. Подбирать наилучшую модель не требуется. 


## Навыки и инструменты

- **python**
- **pandas**
- **numpy**
- **scikit-learn**


## Вывод

В проекте необходимо было защитить данные клиентов страховой компании. Был разработан метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию. Была обоснвана корректность его работы.

Для достижения этих целей было сделано:
1. загрузка необходимых библиотек и исходный датафрейм. В ходе изучения датафлейма некоторые столбцы были преобразованы в необходимый тип данных, а именно: "Возраст" и "Зарплата" в целочисленынй тип данных. Также были удалены дупликаты.
2. Алгебраитечким путем было обосновано, что  умножив признаки на обратиму матрицу предсказания не изменяютсятся. Матрица a1 равняется а. Однако вектор весов линейной регрессии будут различаться.

$$
a_1 = X1w_1 = XPw_1 = XPP^{-1}w = Xw = a
$$

3. Была проведена проверка на числах и доказано, что формула выше верна
4. Обучена модель Линейная регрессия до шифрования и после. Метркиа R2_Score сошлась в обоих случаях. Было показано, что шифрование не влияет на обучение модели.

