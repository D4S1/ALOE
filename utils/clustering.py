import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def _cluster_latent_leiden(
    latent: np.ndarray,
    n_neighbors: int = 20,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """High-resolution Leiden clustering on raw VAE latent space.

    Parameters
    ----------
    latent      : (n_cells, latent_dim) float array
    n_neighbors : k for kNN graph construction
    resolution  : Leiden resolution — higher → more, smaller hyperclusters
    seed        : reproducibility

    Returns
    -------
    clusters : (n_cells,) int array of hypercluster assignments
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(latent)
    knn_graph = nn.kneighbors_graph(latent, mode="connectivity")
    sources, targets = knn_graph.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(edges=edges, directed=False)

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.array(partition.membership)


def _aggregate_vaf_per_hypercluster(
    counts: np.ndarray,
    hypercluster_ids: np.ndarray,
    min_cells: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate SNV counts into per-hypercluster VAF vectors,
    then broadcast each representative back to its member cells.

    Parameters
    ----------
    counts          : (n_cells, n_snvs, 2) array — alt counts at index 0.
    hypercluster_ids: (n_cells,) int array from Leiden
    min_cells       : hyperclusters smaller than this are flagged as unreliable

    Returns
    -------
    vaf_matrix      : (n_hc, n_snvs)    — representative VAF per hypercluster
    hc_labels       : (n_hc,)           — hypercluster ids, same row order as vaf_matrix
    reliable_mask   : (n_hc,)  bool     — True if hypercluster had >= min_cells cells
    cell_vaf_matrix : (n_cells, n_snvs) — each cell replaced by its hypercluster VAF
    """
    unique_hcs = np.unique(hypercluster_ids)
    n_hc        = len(unique_hcs)
    n_cells     = counts.shape[0]
    n_snvs      = counts.shape[1]

    vaf_matrix      = np.zeros((n_hc,    n_snvs), dtype=np.float32)
    cell_vaf_matrix = np.zeros((n_cells, n_snvs), dtype=np.float32)
    reliable_mask   = np.ones(n_hc, dtype=bool)

    for i, hc_id in enumerate(unique_hcs):
        cell_mask    = hypercluster_ids == hc_id
        cell_indices = np.where(cell_mask)[0]

        if len(cell_indices) < min_cells:
            reliable_mask[i] = False

        hc_counts  = counts[cell_indices]         # (n_cells_in_hc, n_snvs, 2)
        summed     = hc_counts.sum(axis=0)         # (n_snvs, 2)
        depth      = summed.sum(axis=-1)           # (n_snvs,)
        safe_depth = np.where(depth == 0, 1, depth)

        rep_vaf = np.where(depth == 0, 0.25, summed[..., 0] / safe_depth)

        vaf_matrix[i]              = rep_vaf
        cell_vaf_matrix[cell_indices] = rep_vaf

    return vaf_matrix, unique_hcs, reliable_mask, cell_vaf_matrix


def _auto_cut_dendrogram(Z: np.ndarray, max_clusters: int = 8) -> np.ndarray:
    """
    Cut a linkage matrix at the largest merge-distance gap.

    Parameters
    ----------
    Z            : linkage matrix from scipy.cluster.hierarchy.linkage
    max_clusters : hard cap on number of clusters returned

    Returns
    -------
    labels : (n,) int array of cluster assignments (1-indexed, as returned by fcluster)
    """
    merge_distances = Z[:, 2]
    gaps = np.diff(merge_distances)
    n = len(Z) + 1

    candidate_cuts = []
    for i, gap in enumerate(gaps):
        n_clusters_if_cut_here = n - (i + 1)
        if 2 <= n_clusters_if_cut_here <= max_clusters:
            candidate_cuts.append((gap, i + 1))

    if not candidate_cuts:
        return np.ones(n, dtype=int)

    best_merge_idx = max(candidate_cuts, key=lambda x: x[0])[1]
    threshold = merge_distances[best_merge_idx] + 1e-10

    return fcluster(Z, t=threshold, criterion="distance")


def _cluster_vaf_hierarchical(
    vaf_matrix: np.ndarray,
    max_clones: int = 8,
    linkage_method: str = "average",
    metric: str = "cityblock",
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Hierarchical clustering of VAF vectors → clone assignments.

    Parameters
    ----------
    vaf_matrix     : (n_cells, n_snvs) — cell_vaf_matrix from aggregate_vaf_per_hypercluster
    max_clones     : ceiling for auto-cut dendrogram
    linkage_method : scipy linkage method
    metric         : pdist metric — 'jaccard' triggers binarization at threshold
    threshold      : binarization cutoff when metric='jaccard'

    Returns
    -------
    labels : (n_cells,) int array of clone assignments (1-indexed)
    """
    if linkage_method == "ward" and metric != "euclidean":
        metric = "euclidean"

    if metric == "jaccard":
        vaf_matrix = np.where(vaf_matrix < threshold, 0, 1)

    dist_vec = pdist(vaf_matrix, metric=metric)
    dist_vec = np.nan_to_num(dist_vec, nan=0.0)

    Z = linkage(dist_vec, method=linkage_method)

    return _auto_cut_dendrogram(Z, max_clusters=max_clones)  # 1-indexed


def cluster_cells(
    latent: np.ndarray,
    counts: np.ndarray,
    label: str,
    leiden_neighbors: int = 15,
    leiden_resolution: float = 1.0,
    min_cells_per_hc: int = 10,
    max_clones: int = 8,
    linkage_method: str = "average",
    metric: str = "cityblock",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Full two-stage clustering pipeline.

    Stage 1 — Leiden on raw latent z → hypercluster_id per cell
    Stage 2 — Hierarchical on VAF representatives → clone_id per cell

    Parameters
    ----------
    latent            : (n_cells, latent_dim)
    counts            : (n_cells, n_snvs, 2)  alt counts at [..., 0]
    label             : column name prefix in output DataFrame
    leiden_neighbors  : kNN graph k
    leiden_resolution : Leiden resolution — higher → more hyperclusters
    min_cells_per_hc  : minimum cells for a hypercluster to be flagged reliable
    max_clones        : ceiling for auto-cut dendrogram
    linkage_method    : scipy linkage method
    metric            : pdist metric for VAF distance
    seed              : for Leiden reproducibility

    Returns
    -------
    DataFrame with columns: ['{label}_hypercluster', '{label}_clone']
    index aligns with input cell order
    """
    # --- Stage 1: hypercluster assignment ---
    hypercluster_ids = _cluster_latent_leiden(
        latent,
        n_neighbors=leiden_neighbors,
        resolution=leiden_resolution,
        seed=seed,
    )

    # --- Aggregation: VAF per hypercluster, broadcast to cells ---
    vaf_matrix, hc_labels, reliable_mask, cell_vaf_matrix = _aggregate_vaf_per_hypercluster(
        counts,
        hypercluster_ids,
        min_cells=min_cells_per_hc,
    )

    # --- Stage 2: clone assignment, size-weighted via cell_vaf_matrix ---
    cell_clone_cityblock = _cluster_vaf_hierarchical(
        cell_vaf_matrix,
        max_clones=max_clones,
        linkage_method=linkage_method,
        metric="cityblock",
    )
    cell_clone_jaccard = _cluster_vaf_hierarchical(
        cell_vaf_matrix,
        max_clones=max_clones,
        linkage_method=linkage_method,
        metric="jaccard",
    )

    return pd.DataFrame({
        f"{label}_hypercluster": hypercluster_ids,
        f"{label}_cityblock_clone": cell_clone_cityblock,
        f"{label}_jaccard_clone": cell_clone_jaccard
        })



