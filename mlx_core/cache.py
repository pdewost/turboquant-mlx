import mlx.core as mx
from .mlx_turboquant import MLXTurboQuant

class TurboQuantKVCache:
    """
    KVCache реализация для Apple MLX.
    Заменяет стандартный mlx_lm.models.cache.KVCache на нашу сжатую версию TurboQuant.
    Она сжимает ключи (и значения по желанию) во время префил-фазы генерации.
    """
    def __init__(self, head_dim: int, n_kv_heads: int, pq_bits: int = 3, qjl_features: int = 2048):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        
        # Для Keys (ключей) важен точный Dot-Product во время Attention, поэтому берем весь конвейер TurboQuant
        self.k_compressor = MLXTurboQuant(feature_dim=head_dim, pq_bits=pq_bits, qjl_features=qjl_features)
        
        # Для Values (значений) важна только минимальная ошибка MSE (Scalar Retrieval), 
        # поэтому здесь QJL (остаток) избыточен. Берем голый PolarQuant.
        from .mlx_polarquant import MLXPolarQuantCompressor
        self.v_compressor = MLXPolarQuantCompressor(feature_dim=head_dim, bits=pq_bits)
        
        self.offset = 0
        self.chunk_size = 64
        
        self.compressed_keys_chunks = []
        self.compressed_values_chunks = []
        
        self.key_buffer = None
        self.value_buffer = None

    def _compress_and_store(self, k: mx.array, v: mx.array):
        b, h, s, d = k.shape
        
        # Сжимаем ключи через TurboQuant
        k_2d = mx.reshape(k, (-1, d))
        compressed_k = self.k_compressor.compress(k_2d)
        self.compressed_keys_chunks.append((compressed_k, (b, h, s, d)))
        
        # Сжимаем значения через PolarQuant (экономим время и ресурсы)
        v_2d = mx.reshape(v, (-1, d))
        compressed_v = self.v_compressor.compress(v_2d)
        self.compressed_values_chunks.append((compressed_v, (b, h, s, d)))

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.offset += keys.shape[2]
        
        if self.key_buffer is None:
            self.key_buffer = keys
            self.value_buffer = values
        else:
            self.key_buffer = mx.concatenate([self.key_buffer, keys], axis=2)
            self.value_buffer = mx.concatenate([self.value_buffer, values], axis=2)
            
        while self.key_buffer is not None and self.key_buffer.shape[2] >= self.chunk_size:
            chunk_k = self.key_buffer[:, :, :self.chunk_size, :]
            chunk_v = self.value_buffer[:, :, :self.chunk_size, :]
            
            self._compress_and_store(chunk_k, chunk_v)
            
            if self.key_buffer.shape[2] > self.chunk_size:
                self.key_buffer = self.key_buffer[:, :, self.chunk_size:, :]
                self.value_buffer = self.value_buffer[:, :, self.chunk_size:, :]
            else:
                self.key_buffer = None
                self.value_buffer = None
                
        # На лету декомпрессируем старые чанки
        full_keys = []
        full_values = []
        
        for comp_k, shape in self.compressed_keys_chunks:
            # Для восстановления ключей в float используем базис (PolarQuant), скрытый в decompress()
            k_approx_2d = self.k_compressor.decompress(comp_k)
            full_keys.append(mx.reshape(k_approx_2d, shape))
            
        for comp_v, shape in self.compressed_values_chunks:
            # Декомпрессируем значения напрямую из PolarQuant
            v_approx_2d = self.v_compressor.decompress(comp_v)
            full_values.append(mx.reshape(v_approx_2d, shape))
            
        if self.key_buffer is not None:
            full_keys.append(self.key_buffer)
            full_values.append(self.value_buffer)
            
        if not full_keys:
            return keys, values
            
        return mx.concatenate(full_keys, axis=2), mx.concatenate(full_values, axis=2)

    @property
    def state(self):
        # Поддержка mlx_lm API, чтобы внутренние фреймворки не ломались
        k = self.key_buffer if self.key_buffer is not None else mx.array([])
        v = self.value_buffer if self.value_buffer is not None else mx.array([])
        return k, v

def apply_turboquant_cache(model=None, bits: int = 3, qjl_features: int = 2048):
    """
    Monkey-patch / Hook для интеграции TurboQuant напрямик в любую LLM (Llama, Gemma) на mlx-lm.
    Подменяет сам генератор KVCache внутри фабрики моделей MLX.
    """
    try:
        import mlx_lm.models.cache as cache_module
    except ImportError:
        print("[TurboQuant] Ошибка: mlx-lm не установлен.")
        return
        
    # Создаем прокси-обертку, чтобы пробрасывать настройки компрессора
    class PatchedCache(TurboQuantKVCache):
        def __init__(self, head_dim: int, n_kv_heads: int, **kwargs):
            super().__init__(head_dim=head_dim, n_kv_heads=n_kv_heads, pq_bits=bits, qjl_features=qjl_features)

    # Глобально подменяем класс KVCache в модуле
    cache_module.KVCache = PatchedCache
            
    print(f"[TurboQuant] Глобальный патч установлен: mlx_lm.models.cache.KVCache подменен. Настройки сжатия: {bits} бит.")
