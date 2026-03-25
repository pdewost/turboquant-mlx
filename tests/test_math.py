import numpy as np
from core.qjl import QJLCompressor

def test_qjl_dot_product_approximation():
    np.random.seed(42)
    d = 128
    k = 8192
    
    compressor = QJLCompressor(feature_dim=d, num_features=k)
    
    actual_dots = []
    estimated_dots = []
    
    for _ in range(100):
        x = np.random.randn(d)
        y = np.random.randn(d)
        
        # Сжатие
        x_quant, norm_x = compressor.compress(x)
        
        # Оценка
        est_dot = compressor.estimate_dot(x_quant, norm_x, y)
        actual_dot = np.dot(x, y)
        
        actual_dots.append(actual_dot)
        estimated_dots.append(est_dot)
        
    actuals_arr = np.array(actual_dots)
    est_arr = np.array(estimated_dots)
    
    # Проверяем высокую корреляцию (> 0.95)
    correlation = np.corrcoef(actuals_arr, est_arr)[0, 1]
    assert correlation > 0.95, f"Correlation too low: {correlation}"
    
    # Проверяем отсутствие системного смещения
    mean_diff = np.mean(actuals_arr - est_arr)
    assert abs(mean_diff) < 1.0, f"Bias is too large: {mean_diff}"

def test_qjl_batch_compression():
    d = 64
    k = 1024
    compressor = QJLCompressor(feature_dim=d, num_features=k)
    
    x_batch = np.random.randn(10, d)
    x_quant, norm_x = compressor.compress(x_batch)
    
    assert x_quant.shape == (10, k)
    assert norm_x.shape == (10, 1)

def test_qjl_estimate_batch():
    d = 128
    k = 2048
    compressor = QJLCompressor(feature_dim=d, num_features=k)
    
    x_batch = np.random.randn(5, d)
    y_single = np.random.randn(d)
    
    x_quant, norm_x = compressor.compress(x_batch)
    
    est = compressor.estimate_dot(x_quant, norm_x, y_single)
    assert len(est) == 5
    
    for i in range(5):
        assert np.isfinite(est[i])

def test_qjl_exact_reconstruction_property():
    # Тест на то, что если y == x, то QJL дает корректную оценку (близкую к квадрату длины)
    d = 128
    k = 16384
    x = np.random.randn(d)
    
    compressor = QJLCompressor(feature_dim=d, num_features=k)
    x_quant, norm_x = compressor.compress(x)
    
    # Сам на себя
    est = compressor.estimate_dot(x_quant, norm_x, x)
    actual = np.dot(x, x)
    
    # Слишком большая погрешность при само-проекции (с большим k) недопустима
    assert abs(est - actual) / actual < 0.1, f"Self-dot estimation failed: est {est}, act {actual}"
    
