import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import NMF

# Page configuration
st.set_page_config(page_title="NMF Rank Convergence Comparison", layout="wide")

# Title
st.title("🔶 NMF Rank Convergence Comparison")

# Tabs
tab1, tab2 = st.tabs(["Controls", "Convergence Plot"])

# Controls Tab
with tab1:
    st.header("Parameters")
    # Iterations slider
    conv_iter = st.slider("Max Iterations", 10, 1000, 200, step=10)
    # Learning rate for PGD
    lr = st.slider("Learning Rate (PGD)", 0.0001, 0.1, 0.001, step=0.0001, format="%.4f")
    # Choose ranks to compare
    available_ranks = list(range(1, 26))
    ranks = st.multiselect("Select Ranks (k) to Compare", available_ranks, default=[2, 5, 10])

# Convergence Plot Tab
with tab2:
    if not ranks:
        st.warning("Please select at least one rank in the Controls tab.")
    else:
        # Generate random nonnegative matrix once (size based on max rank)
        size = max(ranks)
        V = np.abs(np.random.rand(size, size))

        # Convergence function (Multiplicative Update)
        def nmf_mu(V, k, max_iter):
            m, n = V.shape
            W = np.abs(np.random.rand(m, k))
            H = np.abs(np.random.rand(k, n))
            errors = []
            for _ in range(max_iter):
                H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
                W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
                errors.append(np.linalg.norm(V - W @ H, 'fro'))
            return errors

        # Plotly figure
        fig = go.Figure()
        for k in sorted(ranks):
            errs = nmf_mu(V, k, conv_iter)
            fig.add_trace(go.Scatter(
                y=errs,
                mode='lines',
                name=f'k={k}'
            ))

        # Layout
        fig.update_layout(
            title='Reconstruction Error vs Iteration for Various Ranks',
            xaxis_title='Iteration',
            yaxis_title='Frobenius Error',
            width=1400,
            height=700,
            legend_title='Rank (k)',
            template='plotly_white'
        )

        # Display
        st.plotly_chart(fig, use_container_width=True)
