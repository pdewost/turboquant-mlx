import mlx.core as mx

def test_mlx_polarquant():
    from mlx_core.mlx_polarquant import MLXPolarQuantCompressor
    d = 128
    b = 10
    compressor = MLXPolarQuantCompressor(feature_dim=d, bits=5)
    
    key1 = mx.random.key(0)
    x = mx.random.normal((b, d), key=key1)
    
    compressed = compressor.compress(x)
    assert compressed["radius"].shape == (b, 1)
    
    x_approx = compressor.decompress(compressed)
    
    mse = mx.mean((x - x_approx)**2).item()
    print(f"PolarQuant MSE (5-bits): {mse}")
    assert mse < 0.05

def test_mlx_turboquant():
    from mlx_core.mlx_turboquant import MLXTurboQuant
    d = 64
    k = 2048
    
    compressor = MLXTurboQuant(feature_dim=d, pq_bits=3, qjl_features=k)
    
    key1, key2 = mx.random.split(mx.random.key(42))
    
    actuals = []
    ests = []
    
    # 20 samples
    for i in range(20):
        key1, key3 = mx.random.split(key1)
        key2, key4 = mx.random.split(key2)
        x = mx.random.normal((d,), key=key3)
        y = mx.random.normal((d,), key=key4)
        
        compressed = compressor.compress(x)
        est = compressor.estimate_dot(compressed, y).item()
        act = mx.matmul(x, y).item()
        
        actuals.append(act)
        ests.append(est)
        
    import numpy as np
    corr = np.corrcoef(actuals, ests)[0, 1]
    print(f"TurboQuant MLX Correlation: {corr}")
    assert corr > 0.95

if __name__ == "__main__":
    test_mlx_polarquant()
    test_mlx_turboquant()
    print("All MLX integration tests passed successfully!")
