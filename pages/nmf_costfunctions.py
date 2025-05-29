# Interactive Plotly heatmap with built-in sliders (Streamlit-compatible)
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import NMF

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

# Perform NMF with different cost functions
def run_nmf(V, cost):
    model = NMF(n_components=rank, init='random', random_state=0, solver='mu', beta_loss=cost, max_iter=300)
    W = model.fit_transform(V)
    H = model.components_
    V_hat = W @ H
    error = np.linalg.norm(V - V_hat, 'fro')
    return error

error_fro = run_nmf(V_noisy, 'frobenius')
error_kl = run_nmf(V_noisy, 'kullback-leibler')

# Prepare frames for interpolation
frames = []
for i in range(steps + 1):
    alpha = i / steps
    V_interp = (1 - alpha) * V_clean + alpha * V_noisy
    frames.append(go.Frame(
        data=[go.Heatmap(z=V_interp, colorscale='Viridis')],
        name=f'{alpha:.2f}'
    ))

# Build figure
fig = go.Figure(
    data=[go.Heatmap(z=V_clean, colorscale='Viridis')],
    frames=frames,
    layout=go.Layout(
        title='Interpolated Heatmap',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.05,
                x=1.15,
                xanchor='right',
                yanchor='top',
                pad=dict(t=0, r=10),
                buttons=[
                    dict(label='Play', method='animate', args=[None, {'frame': {'duration': 100, 'redraw': True}}]),
                    dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}}])
                ]
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method='animate', args=[[fr.name], {'frame': {'duration': 0, 'redraw': True}}], label=fr.name)
                    for fr in frames
                ],
                transition={'duration': 0},
                x=0, y=-0.1,
                currentvalue={'prefix': 'alpha = '}
            )
        ]
    )
)

fig.update_layout(width=700, height=500)

# Display in Streamlit
st.subheader("Interpolated Heatmap")
st.plotly_chart(fig, use_container_width=True)

# Output reconstruction errors
st.markdown("### NMF Reconstruction Errors")
st.write(f"Frobenius loss error: {error_fro:.4f}")
st.write(f"Kullback-Leibler loss error: {error_kl:.4f}")

st.info("Use the built-in slider or play button to interpolate between clean and noisy data.")
