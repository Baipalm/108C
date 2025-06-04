import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import nnls
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st

# Ensure credentials
os.environ.setdefault('SPOTIPY_CLIENT_ID', "b765b3912e444a5dacd47107641b54aa")
os.environ.setdefault('SPOTIPY_CLIENT_SECRET', "a82dc9645179473a831e2592e4c6cc01")

st.set_page_config(page_title="Playlist Genre Approximation")

CSV_URL = "https://raw.githubusercontent.com/Baipalm/108C/main/H_df.csv"

@st.cache_data  # Caches the result to avoid reloading every time
def load_data(url):
    return pd.read_csv(url, index_col=0)

H_df = load_data(CSV_URL)
# Precomputed NMF components matrix H_df (n_components x n_genres).
# For example, H_df could be loaded from a CSV file or defined elsewhere.
# It should have genres as columns, e.g.:
# H_df = pd.read_csv("H_matrix.csv", index_col=0)
# For demonstration, we'll assume H_df is already available in the environment.

# Example placeholder H_df (replace with actual NMF components):
# H_df = pd.DataFrame(...)

def extract_playlist_id(url: str) -> str:
    """Extract the Spotify playlist ID from the given URL or URI."""
    m = re.search(r'playlist/([A-Za-z0-9]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'playlist:([A-Za-z0-9]+)', url)
    if m:
        return m.group(1)
    raise ValueError(f"Invalid Spotify playlist URL/URI: {url}")

def get_genre_counts(sp_client, playlist_id: str) -> Counter:
    """
    Retrieve all tracks in the playlist and count artist genres.
    Returns a Counter mapping genre -> count.
    """
    artist_ids = []
    limit = 100
    offset = 0
    # Fetch playlist items in batches
    while True:
        resp = sp_client.playlist_items(playlist_id, fields='items.track.artists.id,next',
                                        limit=limit, offset=offset)
        items = resp.get('items', [])
        if not items:
            break
        for item in items:
            track = item.get('track')
            if track and 'artists' in track:
                for artist in track['artists']:
                    artist_id = artist['id']
                    if artist_id:
                        artist_ids.append(artist_id)
        if not resp.get('next'):
            break
        offset += limit

    # Remove duplicates to minimize API calls
    unique_ids = list(set(artist_ids))
    genres_map = {}
    # Fetch artist information in batches of 50
    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i:i+50]
        results = sp_client.artists(batch)['artists']
        for art in results:
            genres_map[art['id']] = art.get('genres', [])

    # Count genres across all artists (including duplicates for multiple tracks)
    counts = Counter()
    for aid in artist_ids:
        genres = genres_map.get(aid, [])
        counts.update(genres)
    return counts

def main():
    st.title("🎵 Input Public Spotify Playlist")
    url = st.text_input("Spotify Playlist URL or URI:")
    if st.button("Analyze") and url:
        try:
            # Extract playlist ID
            pid = extract_playlist_id(url)

            # Initialize Spotify client (assumes credentials in environment or Streamlit secrets)
            sp_client = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

            # Get genre counts for the playlist
            genre_counts = get_genre_counts(sp_client, pid)
            if not genre_counts:
                st.warning("No genres found for this playlist.")
                return

            # Ensure H_df is available (meta-genre components x genres)
            # For example, load H_df from a file or assume it's predefined
            # Here, we check if H_df exists in globals:
            global H_df
            if 'H_df' not in globals():
                st.error("Error: NMF components matrix H_df is not defined.")
                return

            # Align playlist genre vector with H_df's columns
            genres = list(H_df.columns)
            v = np.array([genre_counts.get(g, 0) for g in genres], dtype=float)

            # Solve non-negative least squares: H_df.T * w = v
            # H_df is (n_components x n_genres), so H_df.T is (n_genres x n_components)
            A = H_df.values.T  # shape: (n_genres, n_components)
            b = v              # shape: (n_genres,)
            weights, _ = nnls(A, b)
            
            # Display weights with meta-genre labels
            weight_series = pd.Series(weights, index=H_df.index)
            weight_df = weight_series.to_frame(name='Weight').sort_values(by='Weight', ascending=False)
            
            st.subheader("Meta-Genre Component Weights")
            st.dataframe(weight_df.style.format("{:.4f}"))
            
            st.markdown("Your playlist is a given combination of the following genre profiles ")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
