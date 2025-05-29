# Real-time NMF heatmap updates with widget interface
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import ipywidgets as widgets
from IPython.display import display, clear_output

# Generate clean synthetic non-negative data
np.random.seed(42)
V_clean = np.abs(np.random.rand(40, 60))

# Construct matrix with Poisson-like structure to favor KL divergence
W_true = np.abs(np.random.gamma(shape=1.0, scale=1.0, size=(40, 5)))
H_true = np.abs(np.random.gamma(shape=1.0, scale=1.0, size=(5, 60)))
V_noisy = W_true @ H_true

# Add positive noise resembling count data (to favor KL divergence)
V_noisy += np.random.poisson(lam=0.5, size=V_noisy.shape)
V_noisy = np.clip(V_noisy, 1e-10, None)

# Interactive widget setup
def update_heatmap(alpha):
    V_interp = (1 - alpha) * V_clean + alpha * V_noisy
    clear_output(wait=True)
    display(slider)

    plt.figure(figsize=(8, 6))
    plt.imshow(V_interp, aspect='auto', cmap='viridis')
    plt.title(f'Interpolated V (alpha={alpha:.2f})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='Alpha:')
widgets.interact(update_heatmap, alpha=slider)

print("""
Use the slider to smoothly transition between the clean and noisy data.
""")
