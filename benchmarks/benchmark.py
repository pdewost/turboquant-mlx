import time
import numpy as np
import mlx.core as mx

from core.turboquant import TurboQuant as NumPyTurboQuant
from mlx_core.mlx_turboquant import MLXTurboQuant

def run_benchmark():
    # Симуляция размера кэша ключей/значений в LLM (KV Cache)
    d = 128
    b = 32768 # 32K токенов в кэше
    pq_bits = 3
    k = 2048 # для QJL
    
    print(f"=== Бенчмарк TurboQuant (KV-кэш на {b} токенов, с размерностью {d}) ===")
    
    # Датасет
    np.random.seed(42)
    x_np = np.random.randn(b, d).astype(np.float32)
    y_np = np.random.randn(d).astype(np.float32)
    
    print("\n[NumPy / процессоры CPU]")
    np_compressor = NumPyTurboQuant(feature_dim=d, pq_bits=pq_bits, qjl_features=k)
    
    # Warmup
    _ = np_compressor.compress(x_np[:10])
    
    start = time.perf_counter()
    np_compressed = np_compressor.compress(x_np)
    np_compress_time = time.perf_counter() - start
    print(f"Время сжатия (компрессия): {np_compress_time*1000:.2f} мс")
    
    start = time.perf_counter()
    np_est = np_compressor.estimate_dot(np_compressed, y_np)
    np_est_time = time.perf_counter() - start
    print(f"Оценка скалярного произведения (извлечение/scoring): {np_est_time*1000:.2f} мс")
    
    print("\n[MLX / Metal GPU]")
    mlx_compressor = MLXTurboQuant(feature_dim=d, pq_bits=pq_bits, qjl_features=k)
    
    x_mx = mx.array(x_np)
    y_mx = mx.array(y_np)
    
    # Warmup + Trigger lazy evaluation in MLX
    mx.eval(mlx_compressor.compress(x_mx[:10]))
    
    start = time.perf_counter()
    mlx_compressed = mlx_compressor.compress(x_mx)
    # MLX ленивый - поэтому форсируем вычисления
    mx.eval(*mlx_compressed.values())
    mlx_compress_time = time.perf_counter() - start
    print(f"Время сжатия (компрессия): {mlx_compress_time*1000:.2f} мс")
    
    start = time.perf_counter()
    mlx_est = mlx_compressor.estimate_dot(mlx_compressed, y_mx)
    mx.eval(mlx_est)
    mlx_est_time = time.perf_counter() - start
    print(f"Оценка скалярного произведения (извлечение/scoring): {mlx_est_time*1000:.2f} мс")
    
    print("\n=== Результаты ускорения ===")
    print(f"Ускорение Сжатия (MLX быстрее NumPy): {np_compress_time/mlx_compress_time:.2f}x")
    print(f"Ускорение Извлечения (MLX быстрее NumPy): {np_est_time/mlx_est_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
