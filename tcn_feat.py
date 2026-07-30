"""
Stage III — Temporal Convolutional Network (TCN) feature encoding
=================================================================
Paper: "Enhanced inter-frame video forgery detection using convolutional
network and stacking ensemble"
       Multimedia Tools and Applications (2026) 85:497
       https://doi.org/10.1007/s11042-026-21684-x

Transforms variable-length concatenated features F = [D, O, H] into a fixed
64-D representation F_tcn that encodes short- and long-range temporal
dependencies (paper §2.2.3, Table 2).

Architecture (83,520 trainable parameters)
------------------------------------------
- 3 × causal Conv1D (64 filters, kernel=2, ReLU) with BatchNorm / Dropout
- Residual Add connections after the 2nd and 3rd Conv1D blocks
- 2 × LSTM (64 units, return_sequences=True)
- GlobalAveragePooling1D → 64-D vector per video
- Optimiser: Adam  |  Loss: MSE (temporal reconstruction, not classification)

Pipeline order
--------------
1. annotate.py
2. preprocess_extract.py
3. tcn_feat.py             ← you are here
4. ensemble_class.py

Inputs
------
``output/all_preprocess_feat.json`` (or train/test/val equivalents).

Outputs
-------
``output/all_tcn.npy`` (or ``{split}_tcn.npy``) — shape (N_videos, 64).
"""

import json
import os

import numpy as np
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Dropout,
    Add,
    LSTM,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_tcn_feature_extractor(input_shape):
    """
    Build the Conv1D + residual + LSTM encoder described in Table 2.

    Parameters
    ----------
    input_shape : tuple
        ``(max_length, 1)`` — padded feature sequence length × single channel.

    Returns
    -------
    keras.Model
        Untrained encoder that maps a sequence to a 64-D embedding.
        Compiled with Adam + MSE so ``predict`` can run; in the paper protocol
        the network is used as a temporal feature transformer.
    """
    inputs = Input(shape=input_shape)

    # Block 1 — causal Conv1D preserves temporal order (no future leakage)
    x = Conv1D(
        filters=64,
        kernel_size=2,
        padding="causal",
        activation="relu",
        kernel_regularizer=l2(0.01),
    )(inputs)
    x = BatchNormalization()(x)

    # Block 2 — residual branch
    y = Conv1D(
        filters=64,
        kernel_size=2,
        padding="causal",
        activation="relu",
        kernel_regularizer=l2(0.01),
    )(x)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    x = Add()([x, y])  # residual: improves gradient flow / keeps low-level cues

    # Block 3 — second residual branch
    z = Conv1D(
        filters=64,
        kernel_size=2,
        padding="causal",
        activation="relu",
        kernel_regularizer=l2(0.01),
    )(x)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    x = Add()([x, z])

    # LSTM stack — long-range temporal dependencies across the feature series
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)

    # Collapse variable-length sequence → fixed 64-D vector
    x = GlobalAveragePooling1D()(x)

    model = Model(inputs, x)
    # MSE prioritises temporal reconstruction over direct classification
    model.compile(optimizer=Adam(), loss="mse")

    return model


def extract_tcn_features(tcn_model, X):
    """Run the encoder in inference mode to obtain F_tcn embeddings."""
    features = tcn_model.predict(X)
    return features


def pad_features(features, max_length):
    """
    Zero-pad (or truncate) each 1-D feature vector to ``max_length``.

    Required so the Conv1D/LSTM stack can batch videos of different durations.
    """
    padded_features = []
    for f in features:
        if len(f) < max_length:
            padded_f = np.pad(f, (0, max_length - len(f)), "constant")
        else:
            padded_f = f[:max_length]
        padded_features.append(padded_f)
    return np.array(padded_features)


def load_and_pad_features(file_path, max_length):
    """
    Load preprocess JSON, concatenate D∥O∥H per video, pad, and reshape.

    Returns
    -------
    np.ndarray
        Shape ``(N, max_length, 1)`` ready for the TCN encoder.
    """
    with open(file_path, "r") as f:
        features = json.load(f)
    X = []
    for feature_set in features.values():
        # Concatenate edge, optical_flow, hog into one temporal feature stream
        concatenated_features = np.concatenate(
            [np.array(f) for f in feature_set.values()]
        )
        X.append(concatenated_features)
    X = pad_features(X, max_length)
    X = X.reshape(-1, max_length, 1)
    return X


def main():
    """
    Encode preprocess features into 64-D TCN embeddings and save as .npy.

    Keep ``split_data`` consistent with earlier stages.
    """
    config = {"split_data": True}
    print(f"Current value of split_data: {config['split_data']}")

    if config["split_data"]:
        file_path = os.path.join("output", "all_preprocess_feat.json")

        # Determine the global max concatenated length before padding
        with open(file_path, "r") as f:
            features = json.load(f)
        max_length = max(
            [
                len(np.concatenate([np.array(f) for f in feature_set.values()]))
                for feature_set in features.values()
            ]
        )

        X = load_and_pad_features(file_path, max_length)

        tcn_model = build_tcn_feature_extractor((max_length, 1))
        tcn_features = extract_tcn_features(tcn_model, X)

        with open(os.path.join("output", "all_tcn.npy"), "wb") as f:
            np.save(f, tcn_features)
    else:
        datasets = ["train", "test", "val"]
        max_length = 0

        # Shared max length across splits so embeddings stay comparable
        for dataset in datasets:
            file_path = os.path.join("output", f"{dataset}_preprocess_feat.json")
            with open(file_path, "r") as f:
                features = json.load(f)
            all_lengths = [
                len(np.concatenate([np.array(f) for f in feature_set.values()]))
                for feature_set in features.values()
            ]
            max_length = max(max_length, max(all_lengths))

        for dataset in datasets:
            file_path = os.path.join("output", f"{dataset}_preprocess_feat.json")
            X = load_and_pad_features(file_path, max_length)

            tcn_model = build_tcn_feature_extractor((max_length, 1))
            tcn_features = extract_tcn_features(tcn_model, X)

            with open(os.path.join("output", f"{dataset}_tcn.npy"), "wb") as f:
                np.save(f, tcn_features)


if __name__ == "__main__":
    main()
