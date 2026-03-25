import mlx.core as mx
from mlx_lm import load, generate
from mlx_core.cache import apply_turboquant_cache

# Скрипт проверки модели с использованием TurboQuant (Сжатие 3-бит)
def main():
    model_name = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    print(f"Загрузка модели {model_name}...")
    
    # 1. Загрузка классической квантованной модели (веса 4 бита, но обычный кэш - float16)
    try:
        model, tokenizer = load(model_name)
    except Exception as e:
        print(f"Ошибка загрузки модели. Убедитесь, что установлен mlx-lm: pip install mlx-lm")
        print(e)
        return
    
    # 2. Иглоукалывание: Инъекция нашего сверхсжатого кэша
    print("\n[TurboQuant] Внедрение компрессии KVCache (PolarQuant + QJL)...")
    apply_turboquant_cache(model, bits=3)
    
    # 3. Генерируем "Стог сена" (Большой текст)
    needle = "\nСекретный пароль для реактора — 'AppleSiliconM4Turbo'.\n"
    haystack_chunk = "Это обычный текст, заполняющий контекст системы для проверки памяти. "
    
    # Создаем промпт (можно менять множитель для теста реально огромных контекстов)
    haystack = (haystack_chunk * 300) + needle + (haystack_chunk * 100)
    
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nВычлени точный секретный пароль из текста:\n\n{haystack}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nСекретный пароль:"
    
    print(f"\nРазмер промпта: ~{len(haystack)} символов. Запуск генерации...")
    
    # 4. Генерация (Тут `update_and_fetch` начнет жрать чанки кэша и сжимать их)
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=30, 
        verbose=True
    )
    
    if "AppleSiliconM4Turbo" in response:
        print("\n✅ ТЕСТ ПРОЙДЕН: Модель вспомнила точный факт. TurboQuant сработал идеально!")
    else:
        print("\n❌ ТЕСТ ПРОВАЛЕН: Модель поплыла и забыла факт в сжатом кэше.")

if __name__ == "__main__":
    main()
