import numpy as np
import streamlit as st
import plotly.graph_objects as go

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
def nmf_als(V, rank, max_iter=100):
    m, n = V.shape
    W = np.abs(np.random.rand(m, rank))
    H = np.abs(np.random.rand(rank, n))
    errors = []
    for i in range(max_iter):
        H = np.linalg.lstsq(W, V, rcond=None)[0]
        H[H < 0] = 0
        W = np.linalg.lstsq(H.T, V.T, rcond=None)[0].T
        W[W < 0] = 0
        errors.append(np.linalg.norm(V - W @ H, 'fro'))
    return errors

def nmf_mu(V, rank, max_iter=100):
    m, n = V.shape
    W = np.abs(np.random.rand(m, rank))
    H = np.abs(np.random.rand(rank, n))
    errors = []
    for i in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
        errors.append(np.linalg.norm(V - W @ H, 'fro'))
    return errors

def nmf_pgd(V, rank, max_iter=100, lr=0.001):
    m, n = V.shape
    W = np.abs(np.random.rand(m, rank))
    H = np.abs(np.random.rand(rank, n))
    errors = []
    for i in range(max_iter):
        grad_W = W @ H @ H.T - V @ H.T
        grad_H = W.T @ W @ H - W.T @ V
        W -= lr * grad_W
        H -= lr * grad_H
        W[W < 0] = 0
        H[H < 0] = 0
        errors.append(np.linalg.norm(V - W @ H, 'fro'))
    return errors

# Button to trigger computation (reduces initial load)
if st.sidebar.button("Run NMF Convergence"):
    st.write("Running NMF convergence...")
    errors_als = nmf_als(V_conv, rank, conv_iter)
    errors_mu = nmf_mu(V_conv, rank, conv_iter)
    errors_pgd = nmf_pgd(V_conv, rank, conv_iter, lr)

    # Plot convergence errors
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=errors_als, mode='lines', name='ALS'))
    fig2.add_trace(go.Scatter(y=errors_mu, mode='lines', name='Multiplicative Update'))
    fig2.add_trace(go.Scatter(y=errors_pgd, mode='lines', name='Projected Gradient'))
    fig2.update_layout(
        title='NMF Convergence Comparison',
        xaxis_title='Iterations',
        yaxis_title='Frobenius Norm Error',
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Adjust parameters and click 'Run NMF Convergence' to start.")
