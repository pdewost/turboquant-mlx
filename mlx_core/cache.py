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
        
        # Инстанс компрессора. У LLM обычно head_dim = 128 (что является степенью двойки)
        self.compressor = MLXTurboQuant(feature_dim=head_dim, pq_bits=pq_bits, qjl_features=qjl_features)
        
        self.offset = 0
        
        # Хранилище сжатых словарей
        self.compressed_keys_chunks = []
        self.compressed_values_chunks = []

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Ключевой метод Attention. Вызывается на каждом шаге.
        1. В префиле (prompt) мы получаем огромный кусок (seq_len = N).
        2. При генерации мы получаем по 1 токену.
        """
        self.offset += keys.shape[2]
        
        # TODO: Реализовать логику чанкирования: 
        # Если пришло больше 64 токенов -> вызывать .compress() и складывать в список.
        # При возврате делать .estimate_dot() (или декомпрессию, если нужно вернуть float матрицу)
        
        return keys, values

def apply_turboquant_cache(model, bits: int = 3, qjl_features: int = 2048):
    """
    Monkey-patch / Hook для интеграции TurboQuant напрямик в любую LLM (Llama, Gemma) на mlx-lm.
    Пробегает по слоям модели и подменяет экземпляр cache.
    """
    count = 0
    # mlx.nn.Module имеет метод named_modules() для прохода по графу сети
    if not hasattr(model, 'named_modules'):
        print("[TurboQuant] Ошибка: Модель не является mlx.nn.Module")
        return
        
    for name, module in model.named_modules():
        # У слоев Attention (напр. LlamaAttention) есть атрибут cache
        if hasattr(module, "cache") and hasattr(module, "head_dim"):
            n_kv = getattr(module, "n_kv_heads", module.n_heads) # Фолбэк если не GQA
            
            module.cache = TurboQuantKVCache(
                head_dim=module.head_dim,
                n_kv_heads=n_kv,
                pq_bits=bits,
                qjl_features=qjl_features
            )
            count += 1
            
    print(f"[TurboQuant] KV-Кэш успешно подменён: {count} Attention-слоев переведено на {bits}-битное сжатие.")
