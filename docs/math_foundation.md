# Математический Фундамент: QJL, PolarQuant и TurboQuant

Этот документ содержит выжимку "на пальцах" и псевдокод для алгоритмов, описывающих сжатие векторов (KV-кэша) на основе последних статей от Google Research:
- QJL: [arXiv:2406.03482]
- PolarQuant: [arXiv:2502.02617]
- TurboQuant: [arXiv:2504.19874]

## 1. Quantized Johnson-Lindenstrauss (QJL)

### Концепт
Традиционное квантование требует хранения параметров (scale и zero-point) для каждого блока. QJL устраняет этот оверхед:
1. Вектор $x$ умножается на матрицу $R$ со случайными весами (JL-transform).
2. Берётся только **знак** (sign) проекции. Это 1-битное сжатие.
3. Оценка (Asymmetric Estimator): чтобы найти dot-product ($x \cdot y$), где $x$ — это кэш, а $y$ — это запрос (запрос не квантуется, его умножают на $R$ в float).
4. Их модифицированное скалярное произведение даёт несмещенную (unbiased) оценку без необходимости хранить параметры масштабирования.

### Псевдокод
```python
def qjl_compress(x, R):
    projected = numpy.dot(x, R)
    return numpy.sign(projected)  # 1 bit

def qjl_estimate_dot(x_quant, y, R):
    y_projected = numpy.dot(y, R)
    # Вычисление dot-product с поправкой
    return numpy.dot(x_quant, y_projected) * scaling_factor
```

## 2. PolarQuant

### Концепт
Сжатие с помощью полярных координат.
1. Применяется первичная случайная ротация тензоров (Random Preconditioning).
2. Выполняется рекурсивный переход из Декартовых координат в полярные (углы и радиус).
3. Так как распределение углов после прекондишенинга строго сконцентрировано и математически прогнозируемо, **параметры квантования (scale, zero-point) больше не нужны**. Необходима только равномерная сетка квантования для углов (2-3 бита).

### Псевдокод
```python
def polar_quant_compress(x, R_precondition, bits=3):
    x_rot = numpy.dot(x, R_precondition)
    angles, radius = cartesian_to_polar(x_rot)
    
    # Прямое распределение углов в бинарные корзины без нормировки
    angles_quant = uniform_quantize_fixed(angles, bits)
    
    return angles_quant, radius
```

## 3. TurboQuant

### Концепт
Идеальное комбо лучших черт из двух миров, решающее проблему MSE (качество проекции) и смещения Inner Product-а:
1. Для начала данные "размыкаются" случайной ротацией (чтобы сделать дисперсию координат предсказуемой).
2. Применяется оптимальный координатный квантователь (или Polar квантователь) для минимизации среднеквадратичной ошибки (MSE).
3. Любой MSE-оптимальный квантователь вносит математическое "смещение" при вычислении скалярного произведения, что плохо для Attention-а.
4. **Решение (Двухстадийный конвейер):** вычисляем остаток (residual) после MSE-квантователя и применяем для этого остатка 1-битный QJL. QJL выступает как корректор скалярного произведения, обеспечивая нулевое смещение (unbiased estimation) при минимуме затрат.

### Псевдокод
```python
def turboquant_compress(x, R_rot, MSE_quantizer, R_qjl):
    x_rot = numpy.dot(x, R_rot)
    
    # 1 стадия: оптимизация среднеквадратичной ошибки
    x_mse_quant = MSE_quantizer.compress(x_rot)
    x_mse_approx = MSE_quantizer.decompress(x_mse_quant)
    
    # 2 стадия: коррекция скалярного произведения остатка с помощью QJL
    residual = x_rot - x_mse_approx
    residual_1bit = qjl_compress(residual, R_qjl)
    
    return {
        "mse_data": x_mse_quant, 
        "residual_1bit": residual_1bit
    }
```
