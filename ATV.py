import numpy as np
from scipy.ndimage import sobel

def gradient(H):
    """Compute the gradient of H using Sobel operator."""
    Hx = sobel(H, axis=0)
    Hy = sobel(H, axis=1)
    return Hx, Hy

def gradient_norm(Hx, Hy):
    """Compute the norm of the gradient."""
    return np.sqrt(Hx**2 + Hy**2)

def divergence(Hx, Hy):
    """Compute the divergence of a vector field."""
    div_Hx = sobel(Hx, axis=0)
    div_Hy = sobel(Hy, axis=1)
    return div_Hx + div_Hy

def compute_p(H):
    """Compute p(x, y) = 1 + 1 / (1 + |∇H|^2)."""
    Hx, Hy = gradient(H)
    norm_grad = gradient_norm(Hx, Hy)
    return 1 + 1 / (1 + norm_grad**2)

def compute_expression(H):
    """Compute -div(∇H / |∇H|^(2-p)) where p is computed from the first formula."""
    Hx, Hy = gradient(H)
    norm_grad = gradient_norm(Hx, Hy)
    p = compute_p(H)
    factor = norm_grad**(2 - p)
    factor[factor == 0] = np.finfo(float).eps  # Avoid division by zero
    Hx_normalized = Hx / factor
    Hy_normalized = Hy / factor
    return -divergence(Hx_normalized, Hy_normalized)









# # Example usage
# H = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

# Compute p(x, y)
# p = compute_p(H)
# print("p(x, y):")
# print(p)

# Compute the desired expression
# result = compute_expression(H)
# print("Result:")
# print(result)