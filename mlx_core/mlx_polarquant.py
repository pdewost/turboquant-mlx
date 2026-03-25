import mlx.core as mx
import math

class MLXPolarQuantCompressor:
    def __init__(self, feature_dim: int, bits: int = 3, seed: int = 42):
        self.feature_dim = feature_dim
        self.bits = bits
        self.max_idx = (1 << bits) - 1
        
        assert (feature_dim & (feature_dim - 1)) == 0 and feature_dim > 0, "feature_dim must be power of 2"
        
        # QR init (via numpy for simplicity as it's just one-off logic)
        import numpy as np
        np.random.seed(seed)
        H = np.random.randn(feature_dim, feature_dim)
        Q, R = np.linalg.qr(H)
        d = np.diagonal(R)
        np_R = Q * np.sign(d)
        
        self.R = mx.array(np_R, dtype=mx.float32)
        
    def _quantize_angle(self, angle: mx.array, v_min: float, v_max: float) -> mx.array:
        normalized = (angle - v_min) / (v_max - v_min)
        normalized = mx.clip(normalized, 0.0, 1.0)
        quantized = mx.round(normalized * self.max_idx).astype(mx.int8)
        return quantized
        
    def _dequantize_angle(self, q_angle: mx.array, v_min: float, v_max: float) -> mx.array:
        normalized = q_angle.astype(mx.float32) / self.max_idx
        return normalized * (v_max - v_min) + v_min

    def _cartesian_to_polar_recursive(self, x: mx.array):
        current = x
        angles_list = []
        layer = 0
        
        while current.shape[1] > 1:
            even = current[:, 0::2]
            odd = current[:, 1::2]
            
            radius = mx.sqrt(even**2 + odd**2)
            angle = mx.arctan2(odd, even)
            
            if layer == 0:
                q_angle = self._quantize_angle(angle, -math.pi, math.pi)
            else:
                q_angle = self._quantize_angle(angle, 0.0, math.pi / 2.0)
                
            angles_list.append(q_angle)
            current = radius
            layer += 1
            
        return angles_list, current

    def _polar_to_cartesian_recursive(self, angles_list: list, radius: mx.array):
        current = radius
        for layer in range(len(angles_list)-1, -1, -1):
            q_angle = angles_list[layer]
            
            if layer == 0:
                angle = self._dequantize_angle(q_angle, -math.pi, math.pi)
            else:
                angle = self._dequantize_angle(q_angle, 0.0, math.pi / 2.0)
                
            even = current * mx.cos(angle)
            odd = current * mx.sin(angle)
            
            b = current.shape[0]
            dim_half = current.shape[1]
            stacked = mx.stack([even, odd], axis=-1)
            current = mx.reshape(stacked, (b, dim_half * 2))
            
        return current

    def compress(self, x: mx.array) -> dict:
        is_single = x.ndim == 1
        if is_single:
            x_b = mx.reshape(x, (1, -1))
        else:
            x_b = x
            
        rotated = mx.matmul(x_b, self.R)
        angles_list, radius = self._cartesian_to_polar_recursive(rotated)
        
        if is_single:
            angles_list = [a[0] for a in angles_list]
            radius = radius[0, 0]
            
        return {"angles": angles_list, "radius": radius}

    def decompress(self, compressed: dict) -> mx.array:
        angles_list = compressed["angles"]
        radius = compressed["radius"]
        
        # scalars don't have .ndim in the same way, but mx.array scalar has ndim == 0
        is_single = (not isinstance(radius, list) and getattr(radius, 'ndim', -1) == 0) or isinstance(radius, (float, int))
        
        if is_single:
            radius_b = mx.array([[radius]], dtype=mx.float32)
            angles_b = [mx.expand_dims(a, 0) for a in angles_list]
        else:
            radius_b = radius
            angles_b = angles_list
            
        rotated_approx = self._polar_to_cartesian_recursive(angles_b, radius_b)
        
        original_approx = mx.matmul(rotated_approx, self.R.T)
        
        if is_single:
            original_approx = original_approx[0]
            
        return original_approx
