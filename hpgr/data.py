"""Dataset loading utilities.

Each ``.mat`` file holds a single array (field ``trandata``, ``X`` or ``data``)
whose last column is the label (0 = normal, non-zero = anomaly); all remaining
columns are features.
"""
import os

import numpy as np
import scipy.io as sio


def load_dataset(dataset_dir, selected_file):
    """Load a ``.mat`` dataset and split it into features and labels."""
    data_path = os.path.join(dataset_dir, selected_file)
    print(f"\nLoading dataset: {selected_file}")

    mat = sio.loadmat(data_path)
    if 'trandata' in mat:
        data = mat['trandata']
    elif 'X' in mat:
        data = mat['X']
    elif 'data' in mat:
        data = mat['data']
    else:
        raise KeyError(f"No 'X', 'data' or 'trandata' field found in {selected_file}.")

    # The last column is the anomaly label.
    if data.shape[1] > 1:
        X = data[:, :-1]
        labels = data[:, -1].astype(int)
        print("Using the last column as the anomaly label")
    else:
        X = data
        labels = np.zeros(X.shape[0], dtype=int)
        print("Warning: single-column data, no label available; all samples treated as normal (0).")

    labels = labels.astype(int)

    n_anomalies = int(np.sum(labels != 0))
    print(f"Data shape: {X.shape}")
    print(f"Features: {X.shape}, labels: {labels.shape}")
    print(f"Anomalies: {n_anomalies} / {len(labels)} ({n_anomalies / len(labels) * 100:.2f}%)")

    return X, labels
