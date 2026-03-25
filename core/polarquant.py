import numpy as np

class PolarQuantCompressor:
    def __init__(self, feature_dim: int, bits: int = 3, seed: int = 42):
        """
        Компрессор PolarQuant.
        Использует ортогональную случайную ротацию (preconditioning) 
        и рекурсивное преобразование в полярные координаты с квантованием углов.
        
        :param feature_dim: размерность векторов (должна быть степенью двойки)
        :param bits: количество бит для квантования углов
        """
        self.feature_dim = feature_dim
        self.bits = bits
        self.max_idx = (1 << bits) - 1
        
        # Проверяем что dim - степень двойки
        assert (feature_dim & (feature_dim - 1)) == 0 and feature_dim > 0, "feature_dim должен быть степенью 2"
        
        np.random.seed(seed)
        # Генерация случайной ортогональной матрицы для Preconditioning
        # QR-разложение позволяет получить ортогональную матрицу
        H = np.random.randn(feature_dim, feature_dim)
        Q, R = np.linalg.qr(H)
        d = np.diagonal(R)
        self.R = Q * np.sign(d)
        
    def _quantize_angle(self, angle: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        # Линейное масштабирование в [0, 1] -> в int [0, 2^b - 1]
        normalized = (angle - v_min) / (v_max - v_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        quantized = np.round(normalized * self.max_idx).astype(np.int8)
        return quantized
        
    def _dequantize_angle(self, q_angle: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        normalized = q_angle.astype(np.float32) / self.max_idx
        return normalized * (v_max - v_min) + v_min

    def _cartesian_to_polar_recursive(self, x: np.ndarray):
        """
        Рекурсивный переход батча x в полярные координаты.
        x: (batch, dim)
        Возвращает список массивов углов и финальный массив радиусов (batch, 1).
        """
        current = x
        angles_list = []
        layer = 0
        
        while current.shape[1] > 1:
            even = current[:, 0::2]
            odd = current[:, 1::2]
            
            radius = np.sqrt(even**2 + odd**2)
            angle = np.arctan2(odd, even)
            
            if layer == 0:
                # Оригинальные векторы лежат в [-pi, pi]
                q_angle = self._quantize_angle(angle, -np.pi, np.pi)
            else:
                # Радиусы >= 0, поэтому арктангенс от двух радиусов лежит в [0, pi/2]
                q_angle = self._quantize_angle(angle, 0.0, np.pi/2)
                
            angles_list.append(q_angle)
            current = radius
            layer += 1
            
        return angles_list, current

    def _polar_to_cartesian_recursive(self, angles_list: list, radius: np.ndarray):
        """
        Восстановление. 
        Вход: список углов, radius (b, 1).
        """
        current = radius
        # Идем от последнего слоя (вершина) к нулевому (листья)
        for layer in range(len(angles_list)-1, -1, -1):
            q_angle = angles_list[layer]
            
            # Деквантование
            if layer == 0:
                angle = self._dequantize_angle(q_angle, -np.pi, np.pi)
            else:
                angle = self._dequantize_angle(q_angle, 0.0, np.pi/2)
                
            even = current * np.cos(angle)
            odd = current * np.sin(angle)
            
            # Чередуем элементы: even, odd, even, odd
            b, dim = current.shape
            next_current = np.empty((b, dim * 2), dtype=np.float32)
            next_current[:, 0::2] = even
            next_current[:, 1::2] = odd
            current = next_current
            
        return current

    def compress(self, x: np.ndarray) -> dict:
        """
        Сжатие батча или одиночного вектора.
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)
            
        # Ротация ортогональной матрицей (preconditioning)
        rotated = np.dot(x, self.R)
        
        # Получение квантованных углов и корня (радиуса)
        angles_list, radius = self._cartesian_to_polar_recursive(rotated)
        
        if is_single:
            angles_list = [a[0] for a in angles_list]
            radius = radius[0, 0]
            
        return {"angles": angles_list, "radius": radius}

    def decompress(self, compressed: dict) -> np.ndarray:
        """
        Расжатие (восстановление аппроксимации начального вектора).
        """
        angles_list = compressed["angles"]
        radius = compressed["radius"]
        
        # Проверяем одиночный или батч
        is_single = np.isscalar(radius) or (isinstance(radius, np.ndarray) and radius.ndim == 0)
        
        if is_single:
            radius_b = np.array([[radius]], dtype=np.float32)
            angles_b = [np.expand_dims(a, 0) for a in angles_list]
        else:
            radius_b = radius
            angles_b = angles_list
            
        # Обратное полярное преобразование
        rotated_approx = self._polar_to_cartesian_recursive(angles_b, radius_b)
        
        # Обратная ротация R^T (т.к. матрица ортогональна, R^-1 = R^T)
        original_approx = np.dot(rotated_approx, self.R.T)
        
        if is_single:
            original_approx = original_approx[0]
            
        return original_approx
