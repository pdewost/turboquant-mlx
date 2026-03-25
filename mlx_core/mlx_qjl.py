import mlx.core as mx
import math

class MLXQJLCompressor:
    def __init__(self, feature_dim: int, num_features: int, seed: int = 42):
        self.feature_dim = feature_dim
        self.num_features = num_features
        key = mx.random.key(seed)
        self.R = mx.random.normal((feature_dim, num_features), key=key)
        
    def compress(self, x: mx.array):
        if x.ndim == 1:
            norm_x = mx.linalg.norm(x)
            projected = mx.matmul(x, self.R)
        else:
            norm_x = mx.linalg.norm(x, axis=1, keepdims=True)
            projected = mx.matmul(x, self.R)
            
        x_quant = mx.sign(projected)
        x_quant = mx.where(x_quant == 0.0, mx.array(1.0), x_quant)
        
        return x_quant, norm_x
        
    def estimate_dot(self, x_quant: mx.array, norm_x: mx.array, y: mx.array) -> mx.array:
        y_proj = mx.matmul(y, self.R)
        
        if x_quant.ndim == 2 and y_proj.ndim == 1:
            dot_product = mx.matmul(x_quant, y_proj)
        elif x_quant.ndim == 1 and y_proj.ndim == 2:
            dot_product = mx.matmul(y_proj, x_quant)
        elif x_quant.ndim == 1 and y_proj.ndim == 1:
            dot_product = mx.matmul(x_quant, y_proj)
        else:
            # (b, k) @ (m, k).T -> (b, m)
            dot_product = mx.matmul(x_quant, y_proj.astype(x_quant.dtype).T)
            
        scaling_factor = (norm_x / self.num_features) * math.sqrt(math.pi / 2)
        
        if dot_product.ndim == 1 and getattr(scaling_factor, 'ndim', 0) > 1:
            scaling_factor = mx.squeeze(scaling_factor)
            
        return dot_product * scaling_factor
