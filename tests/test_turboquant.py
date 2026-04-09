import mlx.core as mx
import numpy as np
from core.turboquant import TurboQuant

def test_turboquant_dot_product():
    mx.random.seed(42)
    d = 128
    
    # 2 bits for PQ, 4096 features for QJL
    compressor = TurboQuant(feature_dim=d, pq_bits=2, qjl_features=4096)
    
    actual_dots = []
    estimated_dots = []
    
    for _ in range(50):
        x = mx.random.normal((d,))
        y = mx.random.normal((d,))
        
        compressed = compressor.compress(x)
        est_dot = compressor.estimate_dot(compressed, y)
        act_dot = (x * y).sum()
        
        actual_dots.append(float(act_dot))
        estimated_dots.append(float(est_dot))
        
    # Correlation check using numpy (fine for testing stats)
    corr = np.corrcoef(actual_dots, estimated_dots)[0, 1]
    assert corr > 0.98, f"TurboQuant correlation too low: {corr}"
    
    # Check for systematic bias
    bias = np.mean(np.array(actual_dots) - np.array(estimated_dots))
    assert abs(bias) < 0.5, f"High bias in TurboQuant estimator: {bias}"

def test_turboquant_batch():
    mx.random.seed(42)
    d = 64
    b = 10
    compressor = TurboQuant(feature_dim=d, pq_bits=3, qjl_features=1024)
    
    x_batch = mx.random.normal((b, d))
    y_single = mx.random.normal((d,))
    
    compressed = compressor.compress(x_batch)
    
    assert compressed["qjl_data"].shape == (b, 1024)
    assert compressed["qjl_norm"].shape == (b, 1)
    
    est_dots = compressor.estimate_dot(compressed, y_single)
    assert est_dots.shape == (b,)
    
    for i in range(b):
        assert np.isfinite(float(est_dots[i]))
