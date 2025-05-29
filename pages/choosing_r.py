# streamlit_nmf_rank_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import NMF
import time

st.set_page_config(page_title="NMF Rank Sensitivity", layout="centered")
st.title("Effect of Rank on NMF Reconstruction Error")

# User input for matrix dimensions and true rank
matrix_rows = st.slider("Number of Rows", 10, 200, 50, step=10)
matrix_cols = st.slider("Number of Columns", 10, 200, 80, step=10)
true_rank = st.slider("True Rank (used to generate data)", 1, 20, 5)
noise_level = st.slider("Noise Level", 0.0, 0.5, 0.05, step=0.01)
beta_loss = st.selectbox("Loss Function", ['frobenius', 'kullback-leibler'])

max_iter = st.slider("Max Iterations", 100, 1000, 300, step=100)

if st.button("Run Rank Comparison"):
    np.random.seed(42)
    # Generate synthetic matrix with known low rank
    W_true = np.abs(np.random.rand(matrix_rows, true_rank))
    H_true = np.abs(np.random.rand(true_rank, matrix_cols))
    V = W_true @ H_true + np.random.normal(loc=0.0, scale=noise_level, size=(matrix_rows, matrix_cols))
    V = np.clip(V, 0, None)

    ranks = list(range(1, min(matrix_rows, matrix_cols, 25)))
    errors = []
    durations = []

    for r in ranks:
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

    st.info("""
    Choosing the right rank is crucial:
    - Too low: underfitting, large error.
    - Too high: overfitting, unnecessary complexity and increased time.
    - The best rank balances low error and model simplicity.
    """)
