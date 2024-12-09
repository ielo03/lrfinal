import json
import pandas as pd
from collections import Counter

def count_tracks(column, song_counts=None, total_songs=0):
    """
    Processes a Series containing lists of songs and updates:
    1. The count of each unique song.
    2. The total number of songs.

    Args:
    column (pd.Series): The column containing the song lists.
    song_counts (Counter): Existing Counter object with song counts (default: None).
    total_songs (int): The ongoing total number of songs (default: 0).

    Returns:
    Counter: Updated Counter with the count of each unique song.
    int: Updated total number of songs.
    """
    # Initialize song_counts if not provided
    if song_counts is None:
        song_counts = Counter()

    # Flatten the lists of tracks into a single list
    all_tracks = [track for playlist in column for track in playlist]

    # Update song counts
    song_counts.update(all_tracks)

    # Update the total number of songs
    total_songs += len(all_tracks)

    return song_counts, total_songs

def process_slice_to_df(file_path, df=None):
    """
    Processes a slice of the Spotify dataset into a DataFrame, transforming tracks into an array of strings.

    Args:
        file_path (str): Path to the JSON file containing the slice.
        df (pd.DataFrame, optional): Existing DataFrame to append data to. Defaults to None.

    Returns:
        pd.DataFrame: Updated DataFrame with the new slice's data.
    """
    import numpy as np
    import pandas as pd
    import json

    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract the playlists from the slice
    playlists = data['playlists']

    # Transform tracks into an array of strings `${track_name} by ${artist_name}`
    for playlist in playlists:
        playlist['tracks'] = [f"{track['track_name']} by {track['artist_name']}" for track in playlist['tracks']]

    # Convert playlists to a DataFrame
    slice_df = pd.DataFrame(playlists)

    del slice_df['pid']
    del slice_df['name']
    slice_df['description'] = slice_df['description'].notna()
    slice_df['collaborative'] = slice_df['collaborative'].map({'true': True, 'false': False})
    slice_df['modified_at'] = pd.to_numeric(slice_df['modified_at'], downcast='unsigned')
    slice_df['num_tracks'] = pd.to_numeric(slice_df['num_tracks'], downcast='unsigned')
    slice_df['num_albums'] = pd.to_numeric(slice_df['num_albums'], downcast='unsigned')
    slice_df['num_followers'] = pd.to_numeric(slice_df['num_followers'], downcast='unsigned')
    slice_df['num_edits'] = pd.to_numeric(slice_df['num_edits'], downcast='unsigned')
    slice_df['duration_ms'] = pd.to_numeric(slice_df['duration_ms'], downcast='unsigned')
    slice_df['num_artists'] = pd.to_numeric(slice_df['num_artists'], downcast='unsigned')

    # Add average features
    slice_df['avg_albums_per_artist'] = pd.to_numeric(
        slice_df['num_albums'] / slice_df['num_artists'], downcast='float'
    )
    slice_df['avg_tracks_per_album'] = pd.to_numeric(
        slice_df['num_tracks'] / slice_df['num_albums'], downcast='float'
    )
    slice_df['avg_duration_per_track'] = pd.to_numeric(
        slice_df['duration_ms'] / slice_df['num_tracks'], downcast='float'
    )

    # Drop redundant features
    slice_df = slice_df.drop(columns=['duration_ms', 'num_tracks', 'num_albums'])

    # If no DataFrame is provided, initialize one with the slice data
    if df is None:
        df = slice_df
    else:
        # Append the new slice data to the existing DataFrame
        df = pd.concat([df, slice_df], ignore_index=True)

    return df

import math

def extract_features(df, song_counts, total_songs, top_n=100, decay_rate=0.1):
    """
    Adds features to the playlist Dat aFrame based on track-level data,
    with weights based on song frequency relative to the total number of songs.

    Args:
        df (pd.DataFrame): DataFrame with playlist data.
        song_counts (Counter): A Counter object with song occurrence counts.
        total_songs (int): Total number of songs in the dataset.
        top_n (int): Number of top songs/artists to consider for specific features (default: 100).
        decay_rate (float): Decay rate for exponential weighting based on relative frequency (default: 0.1).

    Returns:
        pd.DataFrame: Updated DataFrame with additional features.
    """
    # Precompute top N songs and artists
    top_songs = {song for song, _ in song_counts.most_common(top_n)}
    top_artists = {track.split(" by ")[-1] for track, _ in song_counts.most_common(top_n)}

    # Precompute exponential weights based on relative frequency
    weights = {}
    for song, count in song_counts.items():
        relative_frequency = count / total_songs  # Frequency as a percentage of total songs
        weights[song] = math.exp(-relative_frequency / decay_rate)

    # Helper functions
    def calculate_popularity_score(tracks):
        """Sum weighted scores of tracks using precomputed weights."""
        return sum(weights.get(track, 0) for track in tracks)

    def count_top_songs(tracks):
        """Count the number of top N songs in the playlist."""
        return sum(1 for track in tracks if track in top_songs)

    def count_top_artists(tracks):
        """Count the number of tracks by top N artists in the playlist."""
        return sum(1 for track in tracks if track.split(" by ")[-1] in top_artists)

    def calculate_diversity_score(tracks):
        """Count unique songs in the playlist."""
        return len(set(tracks))

    def calculate_rarity_score(tracks):
        """Sum rarity scores, inversely proportional to song frequency."""
        return sum(1 - (song_counts.get(track, 0) / total_songs) for track in tracks)

    def unique_artists_count(tracks):
        """Count unique artists in the playlist."""
        return len(set(track.split(" by ")[-1] for track in tracks))

    # Apply feature calculations
    df['popularity_score'] = df['tracks'].apply(calculate_popularity_score)
    df['avg_song_popularity'] = df['popularity_score'] / df['tracks'].apply(len)
    df['top_songs_count'] = df['tracks'].apply(count_top_songs)
    df['top_artists_count'] = df['tracks'].apply(count_top_artists)
    df['diversity_score'] = df['tracks'].apply(calculate_diversity_score)
    df['rarity_score'] = df['tracks'].apply(calculate_rarity_score)
    df['unique_artists'] = df['tracks'].apply(unique_artists_count)

    return df