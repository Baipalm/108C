import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Real-time NMF Convergence", page_icon="📉", layout="wide")

# App title
st.title("🔶 Real-time NMF Convergence")

# Sidebar controls for parameters directly related to convergence
st.sidebar.header("Convergence Parameters")
matrix_size = st.sidebar.slider("Matrix Size (N×N)", 10, 500, 100, step=10)
rank = st.sidebar.slider("Factorization Rank", 1, 50, 5)
conv_iter = st.sidebar.slider("Number of Iterations", 10, 1000, 200, step=10)
learning_rate = st.sidebar.slider("Learning Rate (PGD)", 0.0001, 0.1, 0.001, step=0.0001, format="%.4f")

# Define NMF algorithms to track convergence errors
def nmf_als(V, rank, max_iter):
    m, n = V.shape
    W = np.abs(np.random.rand(m, rank))
    H = np.abs(np.random.rand(rank, n))
    errors = []
    for _ in range(max_iter):
        H = np.linalg.lstsq(W, V, rcond=None)[0]
        H[H < 0] = 0
        W = np.linalg.lstsq(H.T, V.T, rcond=None)[0].T
        W[W < 0] = 0
        errors.append(np.linalg.norm(V - W @ H, 'fro'))
    return errors

def nmf_mu(V, rank, max_iter):
    m, n = V.shape
    W = np.abs(np.random.rand(m, rank))
    H = np.abs(np.random.rand(rank, n))
    errors = []
    for _ in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
        errors.append(np.linalg.norm(V - W @ H, 'fro'))
    return errors

def nmf_pgd(V, rank, max_iter, lr):
    m, n = V.shape
    W = np.abs(np.random.rand(m, rank))
    H = np.abs(np.random.rand(rank, n))
    errors = []
    for _ in range(max_iter):
        grad_W = W @ H @ H.T - V @ H.T
        grad_H = W.T @ W @ H - W.T @ V
        W -= lr * grad_W
        H -= lr * grad_H
        W[W < 0] = 0
        H[H < 0] = 0
        errors.append(np.linalg.norm(V - W @ H, 'fro'))
    return errors

# Generate random nonnegative matrix for convergence test
V = np.abs(np.random.rand(matrix_size, matrix_size))

# Run NMF algorithms to collect convergence errors
errors_als = nmf_als(V, rank, conv_iter)
errors_mu = nmf_mu(V, rank, conv_iter)
errors_pgd = nmf_pgd(V, rank, conv_iter, learning_rate)

# Create plotly figure with all error curves
fig = go.Figure()
fig.add_trace(go.Scatter(y=errors_als, mode='lines', name='ALS'))
fig.add_trace(go.Scatter(y=errors_mu, mode='lines', name='Multiplicative Update'))
fig.add_trace(go.Scatter(y=errors_pgd, mode='lines', name='Projected Gradient'))

fig.update_layout(
    title='NMF Convergence Comparison',
    xaxis_title='Iteration',
    yaxis_title='Frobenius Norm Error',
    width=1200,
    height=600,
    legend_title_text='Algorithm',
    template='plotly_white'
)

# Display the figure full width
st.plotly_chart(fig, use_container_width=True)
