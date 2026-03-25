import numpy as np

class QJLCompressor:
    def __init__(self, feature_dim: int, num_features: int, seed: int = 42):
        """
        Компрессор Quantized Johnson-Lindenstrauss (1-bit KV Cache).
        
        :param feature_dim: размерность исходных векторов (d)
        :param num_features: количество случайных признаков (k) в проекции
        :param seed: seed для воспроизводимости проекции
        """
        self.feature_dim = feature_dim
        self.num_features = num_features
        np.random.seed(seed)
        
        # Случайная матрица проекции
        # Элементы генерируются из N(0, 1)
        self.R = np.random.randn(feature_dim, num_features)
        
    def compress(self, x: np.ndarray):
        """
        Сжимает вектор или батч векторов до 1-битного представления.
        На практике 1 бит (знак) можно упаковывать в int8 (по 8 значений в байт).
        В прототипе мы используем float32/int8 массив для наглядности (со значениями 1 и -1).
        
        :param x: одномерный вектор (d,) или батч векторов (b, d)
        :return: кортеж (x_quant, norm_x)
        """
        if x.ndim == 1:
            norm_x = np.linalg.norm(x)
            projected = np.dot(x, self.R)
        else:
            norm_x = np.linalg.norm(x, axis=1, keepdims=True)
            projected = np.dot(x, self.R)
            
        x_quant = np.sign(projected)
        # Обработка edge-case, если проекция ровно 0
        x_quant[x_quant == 0] = 1.0
        
        return x_quant, norm_x
        
    def estimate_dot(self, x_quant: np.ndarray, norm_x, y: np.ndarray) -> np.ndarray:
        """
        Асимметричная оценка скалярного произведения, где один вектор квантован (x), а 
        второй — запрос из attention без квантования (y).
        
        :param x_quant: квантованный вектор признаков со значениями {-1, 1}
        :param norm_x: L2 норма или вектор норм батча
        :param y: вектор запроса (float)
        """
        y_proj = np.dot(y, self.R)
        
        # Прямое умножение матриц.
        # Если x_quant (b, k) и y_proj (k,) -> получаем (b,)
        # Если оба (k,) -> получаем скаляр
        if x_quant.ndim == 2 and y_proj.ndim == 1:
            dot_product = np.dot(x_quant, y_proj)
        else:
            # Для других комбинаций (например (b, k) и (m, k) -> (b, m)
            dot_product = np.dot(x_quant, y_proj.T)
            
        scaling_factor = (norm_x / self.num_features) * np.sqrt(np.pi / 2)
        
        # Обрезаем размерность (b, 1) для нормального бродкаста с (b,)
        if dot_product.ndim == 1 and getattr(scaling_factor, 'ndim', 0) > 1:
            scaling_factor = np.squeeze(scaling_factor)
            
        return dot_product * scaling_factor
