import mlx.core as mx
from mlx_core.cache import TurboQuantKVCache

def test_asymmetric_cache():
    head_dim = 128
    n_kv_heads = 8
    
    # 1. Инициализация асимметричного кэша
    cache = TurboQuantKVCache(head_dim=head_dim, n_kv_heads=n_kv_heads, pq_bits=3, qjl_features=2048)
    
    # 2. Создаем фейковый контекст (batch=1, heads=8, seq=100, dim=128)
    key1 = mx.random.key(42)
    fake_keys = mx.random.normal((1, n_kv_heads, 100, head_dim), key=key1)
    fake_values = mx.random.normal((1, n_kv_heads, 100, head_dim), key=key1)
    
    # 3. Эмуляция генерации (prefill 100 токенов разом)
    ret_keys, ret_values = cache.update_and_fetch(fake_keys, fake_values)
    
    # 4. Проверяем, что кэш чанкировался: 100 токенов -> 1 чанк (64) + 36 в буфере
    assert len(cache.compressed_keys_chunks) == 1
    assert len(cache.compressed_values_chunks) == 1
    assert cache.key_buffer.shape[2] == 36
    
    # 5. Убеждаемся, что форма возвращаемых значений верная
    assert ret_keys.shape == (1, n_kv_heads, 100, head_dim)
    assert ret_values.shape == (1, n_kv_heads, 100, head_dim)
    
    # 6. Убеждаемся, что значения декомпрессируются без ошибок
    val_approx = ret_values[:, :, :64, :]
    val_orig = fake_values[:, :, :64, :]
    
    mse = mx.mean((val_orig - val_approx)**2).item()
    assert mse < 0.1, f"Values MSE too high: {mse}. PolarQuant failed or was applied incorrectly."
    
    print("Asymmetric KVCache tests passed successfully!")
    print(f"Values MSE after 3-bit PolarQuant compression: {mse:.4f}")

if __name__ == "__main__":
    test_asymmetric_cache()
