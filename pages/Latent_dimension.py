import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(page_title="NMF Convergence Comparison", page_icon="r", layout="wide")

# Title
st.title("🔶 NMF Convergence Comparison with Live Updates")

# Sidebar controls
st.sidebar.header("Convergence Parameters")
matrix_size = st.sidebar.slider("Matrix Size (N×N)", 10, 500, 100, step=10)
rank = st.sidebar.slider("Factorization Rank", 1, 50, 5)
conv_iter = st.sidebar.slider("Iterations", 10, 1000, 200, step=10)
lr = st.sidebar.slider("Learning Rate (PGD)", 0.0001, 0.1, 0.001, step=0.0001, format="%.4f")
update_every = st.sidebar.slider("Update Interval", 1, 50, 5)
run = st.sidebar.button("Run Convergence")

# Cache only data generation
@st.cache_data
def generate_matrix(size, seed=42):
    np.random.seed(seed)
    return np.abs(np.random.rand(size, size))

if run:
    # Prepare data and placeholders
    V = generate_matrix(matrix_size)
    ph_chart = st.empty()
    ph_progress = st.sidebar.progress(0)

    # Initialize for each algorithm
    m, n = V.shape
    # ALS
    W_als = np.abs(np.random.rand(m, rank)); H_als = np.abs(np.random.rand(rank, n)); errors_als = []
    # Multiplicative Update
    W_mu = np.abs(np.random.rand(m, rank)); H_mu = np.abs(np.random.rand(rank, n)); errors_mu = []
    # Projected Gradient
    W_pgd = np.abs(np.random.rand(m, rank)); H_pgd = np.abs(np.random.rand(rank, n)); errors_pgd = []

    # Iterative loop with live updates
    for i in range(conv_iter):
        # ALS step
        H_als = np.linalg.lstsq(W_als, V, rcond=None)[0]; H_als[H_als < 0] = 0
        W_als = np.linalg.lstsq(H_als.T, V.T, rcond=None)[0].T; W_als[W_als < 0] = 0
        errors_als.append(np.linalg.norm(V - W_als @ H_als, 'fro'))
        # MU step
        H_mu *= (W_mu.T @ V) / (W_mu.T @ W_mu @ H_mu + 1e-10)
        W_mu *= (V @ H_mu.T) / (W_mu @ H_mu @ H_mu.T + 1e-10)
        errors_mu.append(np.linalg.norm(V - W_mu @ H_mu, 'fro'))
        # PGD step
        grad_W = W_pgd @ H_pgd @ H_pgd.T - V @ H_pgd.T
        grad_H = W_pgd.T @ W_pgd @ H_pgd - W_pgd.T @ V
        W_pgd -= lr * grad_W; H_pgd -= lr * grad_H
        W_pgd[W_pgd < 0] = 0; H_pgd[H_pgd < 0] = 0
        errors_pgd.append(np.linalg.norm(V - W_pgd @ H_pgd, 'fro'))

        # Update plot intermittently
        if (i % update_every == 0) or (i == conv_iter - 1):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=errors_als, mode='lines', name='ALS'))
            fig.add_trace(go.Scatter(y=errors_mu, mode='lines', name='Multiplicative Update'))
            fig.add_trace(go.Scatter(y=errors_pgd, mode='lines', name='Projected Gradient'))
            fig.update_layout(
                title='Iteration: {}'.format(i+1),
                xaxis_title='Iteration', yaxis_title='Frobenius Norm Error',
                width=1200, height=600, legend_title='Algorithm', template='plotly_white'
            )
            ph_chart.plotly_chart(fig, use_container_width=True)
            ph_progress.progress((i+1)/conv_iter)
            time.sleep(0.01)  # small pause for smoothness

    st.sidebar.success("Completed {} iterations".format(conv_iter))
else:
    st.info("Click 'Run Convergence' to begin live updates.")
