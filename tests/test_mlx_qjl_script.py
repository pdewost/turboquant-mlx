import mlx.core as mx

def test_mlx_qjl():
    from mlx_core.mlx_qjl import MLXQJLCompressor
    import math

    d = 128
    k = 4096
    compressor = MLXQJLCompressor(feature_dim=d, num_features=k)
    
    # Generate random arrays
    key1, key2 = mx.random.split(mx.random.key(0))
    x = mx.random.normal((10, d), key=key1)
    y = mx.random.normal((d,), key=key2)
    
    x_quant, norm_x = compressor.compress(x)
    
    est = compressor.estimate_dot(x_quant, norm_x, y)
    
    # Check correctness (shape and finiteness)
    assert est.shape == (10,)
    assert mx.all(mx.isfinite(est)).item()

    print("MLX QJL Test Passed!")

if __name__ == "__main__":
    test_mlx_qjl()
