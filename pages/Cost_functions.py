import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import NMF

# Page config with error icon
st.set_page_config(page_title="Error Heatmap", page_icon="⚠️", layout="centered")

# Parameters
rows, cols, rank = 40, 60, 5
steps = 50  # number of interpolation steps

# Generate data
def generate_data(r, c, k):
    np.random.seed(42)
    V_clean = np.abs(np.random.rand(r, c))
    W_true = np.abs(np.random.gamma(1.0, 1.0, (r, k)))
    H_true = np.abs(np.random.gamma(1.0, 1.0, (k, c)))
    V_noisy = np.clip(W_true @ H_true + np.random.poisson(0.5, (r, c)), 1e-10, None)
    return V_clean, V_noisy

V_clean, V_noisy = generate_data(rows, cols, rank)

# Sidebar slider for alpha
alpha = st.sidebar.slider("Adjust correlated noise (poisson)", 0.0, 1.0, 0.0, step=1.0/steps)
V_interp = (1 - alpha) * V_clean + alpha * V_noisy

# Perform NMF with different cost functions
def run_nmf(V, cost):
    model = NMF(n_components=rank, init='random', random_state=0, solver='mu', beta_loss=cost, max_iter=300)
    W = model.fit_transform(V)
    H = model.components_
    V_hat = W @ H
    error = np.linalg.norm(V - V_hat, cost)
    return error

error_fro = run_nmf(V_interp, 'frobenius')
error_kl = run_nmf(V_interp, 'kullback-leibler')

# Title
st.title("🔶 Reconstruction Error Heatmap")

# Generate figure
fig = go.Figure(
    data=[go.Heatmap(z=V_interp, colorscale='Viridis')],
    layout=go.Layout(width=700, height=500)
)
st.plotly_chart(fig, use_container_width=True)

# Distinct error metrics displayed below the figure
st.markdown("### Reconstruction Errors")
col1, col2 = st.columns(2)
col1.metric("Frobenius Loss", f"{error_fro:.4f}")
col2.metric("KL Divergence", f"{error_kl:.4f}")
