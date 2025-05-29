# streamlit_nmf_costs.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import NMF
import time

st.set_page_config(page_title="NMF Cost Function Comparison", layout="centered")
st.title("NMF: Frobenius Norm vs Kullback-Leibler Divergence")

# Parameters
matrix_rows = st.slider("Number of Rows", min_value=10, max_value=200, value=40, step=10)
matrix_cols = st.slider("Number of Columns", min_value=10, max_value=200, value=60, step=10)
rank = st.slider("Factorization Rank", 1, 20, 5)
noise_level = st.slider("Gaussian Noise Level", 0.0, 0.5, 0.05, step=0.01)
max_iter = st.slider("Max Iterations", 50, 500, 200, step=50)

if st.button("Run Comparison"):
    np.random.seed(42)

    # Clean matrix
    V_clean = np.abs(np.random.rand(matrix_rows, matrix_cols))

    # Noisy matrix
    W_true = np.abs(np.random.rand(matrix_rows, rank))
    H_true = np.abs(np.random.rand(rank, matrix_cols))
    V_noisy = W_true @ H_true + np.random.normal(loc=0.0, scale=noise_level, size=(matrix_rows, matrix_cols))
    V_noisy = np.clip(V_noisy, 0, None)

    def run_nmf(V, beta_loss, solver):
        model = NMF(n_components=rank, init='random', solver=solver,
                    beta_loss=beta_loss, max_iter=max_iter, random_state=42)
        start = time.time()
        W = model.fit_transform(V)
        H = model.components_
        duration = time.time() - start
        if beta_loss == 'frobenius':
            error = np.linalg.norm(V - W @ H, 'fro')
        else:
            error = model.reconstruction_err_
        return W, H, duration, error

    # Run comparisons
    W_fro_clean, H_fro_clean, time_fro_clean, err_fro_clean = run_nmf(V_clean, 'frobenius', 'cd')
    W_kl_clean, H_kl_clean, time_kl_clean, err_kl_clean = run_nmf(V_clean, 'kullback-leibler', 'mu')
    W_fro_noisy, H_fro_noisy, time_fro_noisy, err_fro_noisy = run_nmf(V_noisy, 'frobenius', 'cd')
    W_kl_noisy, H_kl_noisy, time_kl_noisy, err_kl_noisy = run_nmf(V_noisy, 'kullback-leibler', 'mu')

    st.subheader("Results on Clean Data")
    st.write(f"Frobenius Error: {err_fro_clean:.4f}, Time: {time_fro_clean:.2f}s")
    st.write(f"KL Divergence Error: {err_kl_clean:.4f}, Time: {time_kl_clean:.2f}s")

    st.subheader("Results on Noisy Data")
    st.write(f"Frobenius Error: {err_fro_noisy:.4f}, Time: {time_fro_noisy:.2f}s")
    st.write(f"KL Divergence Error: {err_kl_noisy:.4f}, Time: {time_kl_noisy:.2f}s")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0, 0].imshow(V_clean, aspect='auto', cmap='viridis')
    axs[0, 0].set_title('Original Clean')
    axs[0, 1].imshow(W_fro_clean @ H_fro_clean, aspect='auto', cmap='viridis')
    axs[0, 1].set_title('Frobenius (Clean)')
    axs[0, 2].imshow(W_kl_clean @ H_kl_clean, aspect='auto', cmap='viridis')
    axs[0, 2].set_title('KL (Clean)')
    axs[1, 0].imshow(V_noisy, aspect='auto', cmap='viridis')
    axs[1, 0].set_title('Original Noisy')
    axs[1, 1].imshow(W_fro_noisy @ H_fro_noisy, aspect='auto', cmap='viridis')
    axs[1, 1].set_title('Frobenius (Noisy)')
    axs[1, 2].imshow(W_kl_noisy @ H_kl_noisy, aspect='auto', cmap='viridis')
    axs[1, 2].set_title('KL (Noisy)')
    for ax in axs.flat:
        ax.axis('off')
    st.pyplot(fig)

    st.info("""
    Interpretation:
    - Frobenius norm is robust under Gaussian noise (like real-valued measurements).
    - KL divergence is suitable for sparse, count-based data (e.g., audio spectrograms, document-term matrices).
    """)
