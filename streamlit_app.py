import numpy as np
import plotly.graph_objects as go
import streamlit as st

# NMF using three update rules
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

st.title("Interactive NMF Convergence Visualization")

matrix_size = st.slider("Matrix Size (NxN)", min_value=10, max_value=200, value=50, step=10)
rank = st.slider("Factorization Rank", min_value=1, max_value=20, value=5)
max_iter = st.slider("Number of Iterations", min_value=10, max_value=500, value=100, step=10)
lr = st.slider("Learning Rate (PGD only)", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%f")

if st.button("Run NMF"):
    V = np.abs(np.random.rand(matrix_size, matrix_size))
    errors_als = nmf_als(V, rank, max_iter)
    errors_mu = nmf_mu(V, rank, max_iter)
    errors_pgd = nmf_pgd(V, rank, max_iter, lr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=errors_als, mode='lines', name='ALS'))
    fig.add_trace(go.Scatter(y=errors_mu, mode='lines', name='Multiplicative Update'))
    fig.add_trace(go.Scatter(y=errors_pgd, mode='lines', name='Projected Gradient'))
    fig.update_layout(title='NMF Convergence Comparison', xaxis_title='Iterations', yaxis_title='Frobenius Norm Error')
    st.plotly_chart(fig)
