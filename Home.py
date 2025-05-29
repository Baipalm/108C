import streamlit as st

st.set_page_config(
    page_title="NMF Demonstration",
)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import Counter
from sklearn.decomposition import NMF
import streamlit as st

# Ensure credentials
os.environ.setdefault('SPOTIPY_CLIENT_ID', '6e24be3599a74e7e994196fe19ee4751')
os.environ.setdefault('SPOTIPY_CLIENT_SECRET', '68d61a04d1ea4a37a0a843c475847a2a')
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

def extract_playlist_id(url: str) -> str:
    m = re.search(r'playlist/([A-Za-z0-9]+)', url)
    if m: return m.group(1)
    m = re.search(r'playlist:([A-Za-z0-9]+)', url)
    if m: return m.group(1)
    raise ValueError(f"Invalid playlist URL or URI: {url}")

def get_genre_counts(client, pid: str) -> Counter:
    artist_ids, limit, offset = [], 100, 0
    while True:
        resp = client.playlist_items(pid, market='US',
                                     fields='items.track.artists.id,next',
                                     limit=limit, offset=offset)
        items = resp.get('items', [])
        if not items: break
        for it in items:
            track = it.get('track')
            if track and 'artists' in track:
                artist_ids.extend(a['id'] for a in track['artists'])
        if not resp.get('next'): break
        offset += limit
    unique_ids = list(set(artist_ids))
    genres_map = {}
    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i:i+50]
        for art in client.artists(batch)['artists']:
            genres_map[art['id']] = art.get('genres', [])
    counts = Counter()
    for aid in artist_ids:
        counts.update(genres_map.get(aid, []))
    return counts

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float=1e-10) -> float:
    p_safe, q_safe = p+eps, q+eps
    p_norm, q_norm = p_safe / p_safe.sum(), q_safe / q_safe.sum()
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))

def abbreviate(label: str) -> str:
    parts = label.split()
    if len(parts) == 1:
        return label if len(label) <= 10 else label[:10] + '...'
    first = parts[0]
    initials = '.'.join(p[0] for p in parts[1:])
    return f"{first} {initials}."

def plot_genre_bars_and_heatmap(v1_vis, v2_vis, labels, genres_vis):
    N = len(labels)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    im = axes[0].imshow([v1_vis, v2_vis], aspect='auto', cmap='viridis')
    axes[0].set(yticks=[0,1], yticklabels=['Playlist 1','Playlist 2'],
               xticks=range(N), xticklabels=labels)
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].set_title(f"Top {N} Genres Heatmap (abbreviated)")
    fig.colorbar(im, ax=axes[0], label='Count')

    axes[1].bar(range(N), v1_vis)
    axes[1].set(xticks=range(N), xticklabels=labels)
    axes[1].tick_params(axis='x', labelrotation=90)
    axes[1].set_title('Playlist 1 Genre Counts')

    axes[2].bar(range(N), v2_vis)
    axes[2].set(xticks=range(N), xticklabels=labels)
    axes[2].tick_params(axis='x', labelrotation=90)
    axes[2].set_title('Playlist 2 Genre Counts')

    plt.tight_layout()
    st.pyplot(fig)

def plot_h_matrix(H, top_idx, labels):
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(2):
        ax2[i].bar(range(len(labels)), H[i, top_idx], color='skyblue')
        ax2[i].set(xticks=range(len(labels)), xticklabels=labels)
        ax2[i].tick_params(axis='x', labelrotation=45)
        ax2[i].set_title(f'Meta-{i} Loadings')
    fig2.suptitle("Meta-genre Loadings: bar height shows each genre's contribution.")
    plt.tight_layout()
    st.pyplot(fig2)

def main():
    st.title("🎵 Playlist KL Divergence & Meta-Genre Clustering")

    url1 = st.text_input("Enter first Spotify playlist URL or URI:")
    url2 = st.text_input("Enter second Spotify playlist URL or URI:")
    
    if st.button("Analyze") and url1 and url2:
        try:
            pid1 = extract_playlist_id(url1)
            pid2 = extract_playlist_id(url2)
            c1, c2 = get_genre_counts(sp, pid1), get_genre_counts(sp, pid2)
            genres = sorted(set(c1) | set(c2))
            v1 = np.array([c1.get(g,0) for g in genres], float)
            v2 = np.array([c2.get(g,0) for g in genres], float)
            N = 20
            total = v1 + v2
            top_idx = np.argsort(total)[::-1][:N]
            genres_vis = [genres[i] for i in top_idx]
            v1_vis, v2_vis = v1[top_idx], v2[top_idx]
            labels = [abbreviate(g) for g in genres_vis]

            plot_genre_bars_and_heatmap(v1_vis, v2_vis, labels, genres_vis)

            d12, d21 = kl_divergence(v1, v2), kl_divergence(v2, v1)
            dsym = 0.5 * (d12 + d21)
            st.markdown(f"**KL Divergence:**  \
                        D(1||2) = `{d12:.4f}`, D(2||1) = `{d21:.4f}`, Symmetric = `{dsym:.4f}`")

            Vmat = np.vstack([v1, v2])
            nmf = NMF(n_components=2, init='random', random_state=42,
                      beta_loss='kullback-leibler', solver='mu', max_iter=300)
            W = nmf.fit_transform(Vmat)
            H = nmf.components_
            w_df = pd.DataFrame(W, index=['Playlist 1','Playlist 2'], columns=[f'Meta-{i}' for i in range(2)])
            st.dataframe(w_df.style.format("{:.3f}"))

            plot_h_matrix(H, top_idx, labels)

            st.markdown("**Interpretation of W:**\n" +
                        "- Rows = playlists; cols = latent meta-genres.\n" +
                        "- Higher value = stronger alignment with that meta-genre.\n" +
                        "- Cluster by selecting each row's max-activation column.")

            if dsym <= 0.5:
                st.success("🎉 Very similar playlists!")
            elif dsym <= 1.5:
                st.info("👍 Moderately similar playlists.")
            elif dsym <= 3.0:
                st.warning("⚠️ Playlists differ significantly.")
            else:
                st.error("❌ Playlists are very different.")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()

# Notes for NMF application of KLD
# Raw data is high-dimensional (our genre count) vectors (1-Dimension per genre)
# in which NMF factors that matrix V into smaller non-negative matricies W,H 
# W (playlist x meta-genre activation) tells us that for each playlist how strongly it loads on each of the 
# k factors (meta genres)
# H (meta-genre x genere loadings) which describes each meta genre as a distribution over the original genres
# This process compresses the data from large sparse counts into smaller easier to manage counts
# When we get W we have that each playlist is represented as k-dimenstional point in the metagenre space,
# similar playlists have similar W-rows
# We assign each playlist to the meta-genre with highest activation (clustering)
# When we run the program we see that playlists dominated by the same latent
# factor fall into the same cluser (rock heavy cluster vs disco)
# This program shows the divergence between two playlists (KL-Div)
# but also shows the genre structures that tie playlists together
