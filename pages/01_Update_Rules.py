import numpy as np
import streamlit as st
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Real-time NMF Convergence", page_icon="📉", layout="wide")
st.title("Real-time NMF Convergence")

# Sidebar controls
st.sidebar.header("NMF Convergence Settings")
matrix_size = st.sidebar.slider("Matrix Size (NxN)", 10, 100, 50, step=10)
rank = st.sidebar.slider("Factorization Rank", 1, 10, 5)
conv_iter = st.sidebar.slider("Convergence Iterations", 10, 200, 50, step=10)
lr = st.sidebar.slider("Learning Rate (PGD only)", 0.0001, 0.1, 0.001, step=0.0001, format="%f")

# Cache matrix generation to avoid regeneration on every rerun
@st.cache_data
def generate_matrix(size):
    np.random.seed(42)
    return np.abs(np.random.rand(size, size))

V_conv = generate_matrix(matrix_size)

# NMF implementations

def nmf_step_als(V, W, H):
    H = np.linalg.lstsq(W, V, rcond=None)[0]
    H[H < 0] = 0
    W = np.linalg.lstsq(H.T, V.T, rcond=None)[0].T
    W[W < 0] = 0
    return W, H

def nmf_step_mu(V, W, H):
    H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
    W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
    return W, H

def nmf_step_pgd(V, W, H, lr):
    grad_W = W @ H @ H.T - V @ H.T
    grad_H = W.T @ W @ H - W.T @ V
    W -= lr * grad_W
    H -= lr * grad_H
    W[W < 0] = 0
    H[H < 0] = 0
    return W, H

# Button to trigger computation
if st.sidebar.button("Run NMF Convergence"):
    st.write("Running real-time NMF convergence...")

    # Initializations
    m, n = V_conv.shape
    W_als = np.abs(np.random.rand(m, rank))
    H_als = np.abs(np.random.rand(rank, n))
    W_mu = W_als.copy()
    H_mu = H_als.copy()
    W_pgd = W_als.copy()
    H_pgd = H_als.copy()

    errors_als = []
    errors_mu = []
    errors_pgd = []

    graph_placeholder = st.empty()
    error_placeholder = st.empty()

    for i in range(conv_iter):
        W_als, H_als = nmf_step_als(V_conv, W_als, H_als)
        W_mu, H_mu = nmf_step_mu(V_conv, W_mu, H_mu)
        W_pgd, H_pgd = nmf_step_pgd(V_conv, W_pgd, H_pgd, lr)

        if i % 1 == 0:
            err_als = np.linalg.norm(V_conv - W_als @ H_als, 'fro')
            err_mu = np.linalg.norm(V_conv - W_mu @ H_mu, 'fro')
            err_pgd = np.linalg.norm(V_conv - W_pgd @ H_pgd, 'fro')
            errors_als.append(err_als)
            errors_mu.append(err_mu)
            errors_pgd.append(err_pgd)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=errors_als, mode='lines+markers', name='ALS'))
            fig2.add_trace(go.Scatter(y=errors_mu, mode='lines+markers', name='Multiplicative Update'))
            fig2.add_trace(go.Scatter(y=errors_pgd, mode='lines+markers', name='Projected Gradient'))
            fig2.update_layout(
                title='NMF Convergence Comparison (Live)',
                xaxis_title='Iteration',
                yaxis_title='Frobenius Norm Error',
                width=900,
                height=600,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            graph_placeholder.plotly_chart(fig2, use_container_width=True)
            error_placeholder.markdown(f"**Iteration {i+1}**  ")
            error_placeholder.markdown(f"ALS Error: `{err_als:.4f}`  ")
            error_placeholder.markdown(f"MU Error: `{err_mu:.4f}`  ")
            error_placeholder.markdown(f"PGD Error: `{err_pgd:.4f}`")

            time.sleep(0.1)
else:
    st.info("Adjust parameters and click 'Run NMF Convergence' to start.")
