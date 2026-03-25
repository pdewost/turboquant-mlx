import numpy as np
from core.qjl import QJLCompressor
from core.polarquant import PolarQuantCompressor

class TurboQuant:
    def __init__(self, feature_dim: int, pq_bits: int = 3, qjl_features: int = 2048, seed: int = 42):
        """
        Компрессор TurboQuant (Двухстадийный конвейер для KV-Cache).
        
        1. Использует PolarQuant (MSE-оптимальный квантователь).
        2. Вычисляет ошибку (остаток).
        3. Квантует ошибку через QJL, чтобы устранить смещение скалярного произведения (bias).
        
        :param feature_dim: размерность вектора (d), должна быть степенью двойки
        :param pq_bits: битрейт для квантования углов в PolarQuant
        :param qjl_features: количество случайных признаков (k) для QJL 
        """
        self.feature_dim = feature_dim
        
        # Базовый квантователь (минимизирует L2 расстояние)
        self.pq = PolarQuantCompressor(feature_dim=feature_dim, bits=pq_bits, seed=seed)
        
        # 1-битный корректор скалярного произведения (остатка)
        self.qjl = QJLCompressor(feature_dim=feature_dim, num_features=qjl_features, seed=seed+1)
        
    def compress(self, x: np.ndarray) -> dict:
        """
        Двухстадийное сжатие вектора или батча.
        :param x: вектор (d,) или батч (b, d)
        :return: словарь с компрессированными данными
        """
        # --- СТАДИЯ 1: MSE Квантование (PolarQuant) ---
        pq_compressed = self.pq.compress(x)
        
        # Восстанавливаем аппроксимированный вектор
        x_mse_approx = self.pq.decompress(pq_compressed)
        
        # --- СТАДИЯ 2: Остаток (Residual) + QJL ---
        residual = x - x_mse_approx
        
        # Квантуем остаток через QJL до 1 бита (знак) + храня норму L2 (одно число на вектор)
        qjl_quant, qjl_norm = self.qjl.compress(residual)
        
        return {
            "pq_data": pq_compressed,
            "qjl_data": qjl_quant,
            "qjl_norm": qjl_norm
        }
        
    def estimate_dot(self, compressed: dict, y: np.ndarray) -> np.ndarray:
        """
        Несмещенная оценка (unbiased estimation) скалярного произведения.
        :param compressed: сжатые данные от compress()
        :param y: оригинальный несжатый запрос (float-вектор формы (d,))
        :return: оценка x * y
        """
        # 1. Классическое скалярное произведение аппроксимированного вектора
        x_mse_approx = self.pq.decompress(compressed["pq_data"])
        
        # Обрабатываем размерности: батч/одиночка x запросы
        if x_mse_approx.ndim == 2 and y.ndim == 1:
            dot_mse = np.dot(x_mse_approx, y)
        else:
            dot_mse = np.dot(x_mse_approx, y.T)
            
        # 2. Оценка скалярного произведения остатка (компенсирует смещение dot_mse)
        dot_residual = self.qjl.estimate_dot(
            x_quant=compressed["qjl_data"], 
            norm_x=compressed["qjl_norm"], 
            y=y
        )
        
        # Итоговое скалярное произведение = Доля от PolarQuant + Скорректированная доля ошибки
        return dot_mse + dot_residual
