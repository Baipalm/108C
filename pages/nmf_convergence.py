import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import NMF
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Real-time NMF Analysis", layout="centered")
st.title("Real-time NMF Analysis")

# --- Controls ---
st.sidebar.header("Data Generation & NMF Settings")
matrix_rows = st.sidebar.slider("Rows", 10, 200, 50, step=10)
matrix_cols = st.sidebar.slider("Columns", 10, 200, 80, step=10)
true_rank = st.sidebar.slider("True Rank", 1, 20, 5)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.05, step=0.01)
beta_loss = st.sidebar.selectbox("Loss Function", ['frobenius', 'kullback-leibler'])
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, step=100)

# --- Generate synthetic matrix ---
st.subheader("Synthetic Matrix Generation")
np.random.seed(42)
W_true = np.abs(np.random.rand(matrix_rows, true_rank))
H_true = np.abs(np.random.rand(true_rank, matrix_cols))
V = W_true @ H_true + np.random.normal(loc=0.0, scale=noise_level, size=(matrix_rows, matrix_cols))
V = np.clip(V, 0, None)
st.write("Synthetic matrix generated with true rank", true_rank)

# --- Real-time NMF Rank Comparison ---
st.subheader("Rank Comparison")
ranks = list(range(1, min(matrix_rows, matrix_cols, 25)))
errors = []
durations = []
progress_bar = st.progress(0)
status_text = st.empty()

for i, r in enumerate(ranks):
    model = NMF(n_components=r, init='random', solver='cd' if beta_loss == 'frobenius' else 'mu',
                beta_loss=beta_loss, max_iter=max_iter, random_state=42)
    start = time.time()
    W = model.fit_transform(V)
    H = model.components_
    duration = time.time() - start

    if beta_loss == 'frobenius':
        err = np.linalg.norm(V - W @ H, 'fro')
    else:
        err = model.reconstruction_err_

    errors.append(err)
    durations.append(duration)

    progress_bar.progress((i + 1) / len(ranks))
    status_text.text(f"Rank {r}: error={err:.4f}, time={duration:.2f}s")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(ranks, errors, marker='o')
ax[0].set_title("Reconstruction Error vs Rank")
ax[0].set_xlabel("Rank")
ax[0].set_ylabel("Reconstruction Error")

ax[1].plot(ranks, durations, marker='x', color='orange')
ax[1].set_title("Runtime vs Rank")
ax[1].set_xlabel("Rank")
ax[1].set_ylabel("Time (s)")

st.pyplot(fig)

# --- Real-time NMF Convergence ---
st.subheader("Real-time NMF Convergence Visualization")
matrix_size = st.slider("Matrix Size (NxN)", 10, 200, 50, step=10)
rank = st.slider("Factorization Rank", 1, 20, 5)
conv_iter = st.slider("Convergence Iterations", 10, 500, 100, step=10)
lr = st.slider("Learning Rate (PGD only)", 0.0001, 0.1, 0.001, step=0.0001, format="%f")

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

# Generate matrix and show real-time progress
V_conv = np.abs(np.random.rand(matrix_size, matrix_size))
st.write("Convergence matrix generated.")
errors_als = nmf_als(V_conv, rank, conv_iter)
errors_mu = nmf_mu(V_conv, rank, conv_iter)
errors_pgd = nmf_pgd(V_conv, rank, conv_iter, lr)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=errors_als, mode='lines', name='ALS'))
fig2.add_trace(go.Scatter(y=errors_mu, mode='lines', name='Multiplicative Update'))
fig2.add_trace(go.Scatter(y=errors_pgd, mode='lines', name='Projected Gradient'))
fig2.update_layout(title='NMF Convergence Comparison', xaxis_title='Iterations', yaxis_title='Frobenius Norm Error')
st.plotly_chart(fig2)

st.info("""
This dashboard shows how NMF behaves under different ranks and update rules:
- Explore the trade-off between rank, error, and runtime.
- Understand convergence dynamics using ALS, MU, and PGD.
""")
