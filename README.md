# Решение хакатона Raifhack 2021
Публичное решение хакатона Raifhack2021

# Скоры:
LB_score: 1.4289593437149333 (25.09.21 21:10:38)

CV_scores:
1) NN: 1.4365
2) LGB: 1.6616
3) XGB: 1.492
4) Weighted: 1.282784
    
# Краткое описание решения:

1) Считывание данных
2) Хот энкодинг категориальных фичей
3) Применение квантильного преобразования для численных фичей
4) Применение минмакс преобразования для численных фичей
5) Обучение моделей: NN, LGB, XGB
6) Поиск оптимальных весов для модлелей

# Метрика
Для LightGBM и XGBoost была использована функция потерь = метрика задачи. Для NN такая метрика не очень хорошо подходит ввиду нулевой производной в некоторых точках графика. Поэтому была реализована собственная функция потерь (очень похожая на метрику задачи) для NN:

```
def tf_custom_loss(y_true, y_pred):     
    threshold = 0.6     
    error = tf.abs(y_true - y_pred) / y_true
    is_small_error = error <= threshold     
    small_error_loss = tf.square(error / 0.15 - 1)    
    big_error_loss = 9.0 * tf.ones_like(small_error_loss) + tf.abs(error)    
    return tf.where(is_small_error, small_error_loss, big_error_loss)
```
