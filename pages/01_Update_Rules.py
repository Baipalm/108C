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

@st.cache_data
def generate_matrix(size):
    np.random.seed(42)
    return np.abs(np.random.rand(size, size))

V_conv = generate_matrix(matrix_size)

# Function to yield updates per iteration
def nmf_all_live(V, rank, max_iter=100, lr=0.001):
    m, n = V.shape
    W_als = np.abs(np.random.rand(m, rank))
    H_als = np.abs(np.random.rand(rank, n))
    W_mu = np.abs(np.random.rand(m, rank))
    H_mu = np.abs(np.random.rand(rank, n))
    W_pgd = np.abs(np.random.rand(m, rank))
    H_pgd = np.abs(np.random.rand(rank, n))

    errors_als, errors_mu, errors_pgd = [], [], []

    for i in range(max_iter):
        # ALS update
        H_als = np.linalg.lstsq(W_als, V, rcond=None)[0]
        H_als[H_als < 0] = 0
        W_als = np.linalg.lstsq(H_als.T, V.T, rcond=None)[0].T
        W_als[W_als < 0] = 0
        err_als = np.linalg.norm(V - W_als @ H_als, 'fro')
        errors_als.append(err_als)

        # MU update
        H_mu *= (W_mu.T @ V) / (W_mu.T @ W_mu @ H_mu + 1e-10)
        W_mu *= (V @ H_mu.T) / (W_mu @ H_mu @ H_mu.T + 1e-10)
        err_mu = np.linalg.norm(V - W_mu @ H_mu, 'fro')
        errors_mu.append(err_mu)

        # PGD update
        grad_W = W_pgd @ H_pgd @ H_pgd.T - V @ H_pgd.T
        grad_H = W_pgd.T @ W_pgd @ H_pgd - W_pgd.T @ V
        W_pgd -= lr * grad_W
        H_pgd -= lr * grad_H
        W_pgd[W_pgd < 0] = 0
        H_pgd[H_pgd < 0] = 0
        err_pgd = np.linalg.norm(V - W_pgd @ H_pgd, 'fro')
        errors_pgd.append(err_pgd)

        yield i, errors_als, errors_mu, errors_pgd

if st.sidebar.button("Run NMF Convergence"):
    st.write("Streaming real-time NMF convergence...")

    chart = st.empty()
    text_out = st.empty()

    errors_als, errors_mu, errors_pgd = [], [], []

    for i, errors_als, errors_mu, errors_pgd in nmf_all_live(V_conv, rank, conv_iter, lr):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=errors_als, mode='lines', name='ALS'))
        fig2.add_trace(go.Scatter(y=errors_mu, mode='lines', name='Multiplicative Update'))
        fig2.add_trace(go.Scatter(y=errors_pgd, mode='lines', name='Projected Gradient'))
        fig2.update_layout(
            title='NMF Convergence Comparison',
            xaxis_title='Iteration',
            yaxis_title='Frobenius Norm Error',
            width=900,
            height=600,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        chart.plotly_chart(fig2, use_container_width=True)

        text_out.markdown(f"""
        **Iteration {i+1}/{conv_iter}**
        - ALS Error: `{errors_als[-1]:.5f}`
        - MU Error: `{errors_mu[-1]:.5f}`
        - PGD Error: `{errors_pgd[-1]:.5f}`
        """)

        time.sleep(0.1)
else:
    st.info("Adjust parameters and click 'Run NMF Convergence' to start.")
