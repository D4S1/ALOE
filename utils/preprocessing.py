import numpy as np
from sklearn.preprocessing import OneHotEncoder
import Levenshtein
import pandas as pd
import tensorflow as tf


def _get_mapping(
    df: pd.DataFrame,
    col: str,
    threshold: int=0,
    category: str="bcr"
    ) -> dict:
    """
    Group cell annotation based on levenshtein distance between BCR sequences
    """
    unique_values = df[col].unique()

    clusters = []
    visited = set()
    n = len(unique_values)

    for i in range(n):
        if i in visited:
            continue
        cluster = [unique_values[i]]
        visited.add(i)

        for j in range(i + 1, n):
            if j not in visited:
              dist = Levenshtein.distance(unique_values[i], unique_values[j])
              if dist <= threshold:
                  cluster.append(unique_values[j])
                  visited.add(j)
        clusters.append(cluster)

    mapping = {}
    for i, cluster in enumerate(clusters, start=1):
        for seq in cluster:
            mapping[seq] = f"{category}{i}"
    return mapping


def prep_data(
    path: str,
    distance_th: int = 0,
    cluster_th: int = 10,
    val_size: float = 0.2,
    seed: int = 42,
    simulated: bool = False,
    transpose: bool = True
) -> dict:

    assert 0.0 < val_size < 1.0, "val_size must be in (0, 1)"

    snv = pd.read_csv(path + '/A.csv', header=0, index_col=0)
    snv = snv.T if transpose else snv
    snv = snv.to_numpy().astype(np.float32)

    counts = pd.read_csv(path + '/D.csv', header=0, index_col=0)
    counts = counts.T if transpose else counts
    counts = counts.to_numpy().astype(np.float32)

    snv = np.stack([snv, counts - snv], axis=-1)
    n_sample, n_snv = snv.shape[0], snv.shape[1]

    if simulated:
      bcr = np.load(path + '/BCR.npy')
      bcr = tf.convert_to_tensor(bcr, dtype=tf.float32)
      bcr_df = pd.read_csv(path + '/metadata.csv', header=0, index_col=0)
    else:
      bcr_df = pd.read_csv(path + '/heavy_and_light.csv', index_col=0)
      bcr_df['bcr'] = bcr_df['heavy'] + bcr_df['light']
      bcr_mapping = _get_mapping(bcr_df, 'bcr', distance_th)
      bcr_df['bcr_cluster'] = bcr_df['bcr'].map(bcr_mapping)

      cluster_sizes = bcr_df.groupby('bcr_cluster').size()
      small_clusters = cluster_sizes[cluster_sizes < cluster_th].index
      bcr_df['cluster'] = bcr_df['bcr_cluster'].apply(
          lambda x: 'unassign' if x in small_clusters else x
      )

      bcr_df['bcr'] = bcr_df['bcr'].apply(list)
      bcr_array = np.array(bcr_df['bcr'].to_list()).reshape(-1, 1)

      bcr = OneHotEncoder(categories=[['a', 't', 'g', 'c']]) \
          .fit_transform(bcr_array)

      bcr = tf.constant(
          bcr.toarray().reshape(n_sample, -1, 4),
          dtype=tf.float32
      )
      bcr_df = bcr_df[['bcr_cluster', 'cluster']]

    n_bcr = bcr.shape[1]

    # ===== shuffle indices =====
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_sample)

    snv = snv[idx]
    counts = counts[idx]
    bcr = tf.gather(bcr, idx)
    bcr_df = bcr_df.iloc[idx]

    # ===== split =====
    n_val = int(n_sample * val_size)
    n_train = n_sample - n_val

    snv_train, snv_val = snv[:n_train], snv[n_train:]
    counts_train, counts_val = counts[:n_train], counts[n_train:]

    bcr_train, bcr_val = bcr[:n_train], bcr[n_train:]
    dummy_train = tf.math.reduce_sum(bcr_train, axis=-1)
    dummy_val = tf.math.reduce_sum(bcr_val, axis=-1)

    return {
        'n_sample': n_sample,
        'n_snv': n_snv,
        'n_bcr': n_bcr,
        'train': ((snv_train, counts_train), (bcr_train, dummy_train)),
        'val': ((snv_val, counts_val), (bcr_val, dummy_val)),
        'dataset': ((snv, counts), (bcr, tf.math.reduce_sum(bcr, axis=-1)))
    }, bcr_df