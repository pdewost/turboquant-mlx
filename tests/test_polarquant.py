import numpy as np
from core.polarquant import PolarQuantCompressor

def test_polarquant_single_vector():
    d = 128
    bits = 4
    compressor = PolarQuantCompressor(feature_dim=d, bits=bits)
    
    x = np.random.randn(d)
    compressed = compressor.compress(x)
    
    # 1 radius is stored + log2(d) angles arrays
    assert len(compressed["angles"]) == 7 # log2(128) = 7
    assert np.isscalar(compressed["radius"]) or compressed["radius"].ndim == 0
    
    x_approx = compressor.decompress(compressed)
    
    assert x_approx.shape == x.shape
    
    # MSE должен быть приемлемым (по крайней мере < 0.1 для случайного шума с дисперсией 1)
    mse = np.mean((x - x_approx)**2)
    assert mse < 0.1, f"MSE {mse} is too large for 4 bits."

def test_polarquant_batch():
    d = 64
    b = 10
    bits = 5
    compressor = PolarQuantCompressor(feature_dim=d, bits=bits)
    
    x_batch = np.random.randn(b, d)
    compressed = compressor.compress(x_batch)
    
    # Radius shape: (b, 1)
    assert compressed["radius"].shape == (b, 1)
    
    x_approx = compressor.decompress(compressed)
    assert x_approx.shape == x_batch.shape
    
    # Check that mse improves with higher bits (5 bits should be < 0.05)
    mse = np.mean((x_batch - x_approx)**2)
    assert mse < 0.05, f"MSE {mse} is too large for 5 bits."
