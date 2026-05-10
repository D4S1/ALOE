from FACTMx.FACTMx_model import FACTMx_model
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import os
import json

from utils.preprocessing import prep_data
import utils.visualization as viz


heavy_chains = {
    'K4B': 381,
    'K5B': 381,
    'K6B': 360,
    'K7B': 362,
    'synthetic': 350
}


def _smape_error(x, x_hat, eps=1e-8):
    # SMAPE w zakresie [0,1]
    smape = np.abs(x - x_hat) / (np.abs(x) + np.abs(x_hat) + eps)

    # a) per cell -> średnia po snv
    per_cell = smape.mean(axis=1)   # [cells, 2]

    # b) per snv -> średnia po cells
    per_snv = smape.mean(axis=0)    # [snv, 2]

    return smape, per_cell, per_snv


def _map_prediction2ref(
    annotation: pd.DataFrame,
    label: str,
    ref_col: str='clone',
    pred_col: str='clone_hat',
    inplace: bool=False
    ) -> pd.DataFrame:
    """
    Maps clusters to clones based on majority rule.
    """
    mapping = (
        annotation.groupby(label)[ref_col]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    # 2. Assign the 'consensus' cluster to the new column
    annotation = annotation if inplace else annotation.copy()
    annotation[pred_col] = annotation[label].map(mapping)

    return annotation


def _hyper_cluster_ari(
    annotation: pd.DataFrame,
    model_name: str,
    label: str,
    save_dir: str
) -> None:
    pred_labels = annotation[f"{model_name}_hypercluster"]
    true_labels = annotation["hypercluster"]
    ari = adjusted_rand_score(true_labels, pred_labels)

    json_path = os.path.join(save_dir, f"{label}_hypercluster_ari.json")
    with open(json_path, 'w') as f:
        json.dump(ari, f, indent=4)


def _avg_genotype_accuracy(
    annotation: pd.DataFrame,
    counts: np.ndarray,    # (n_cells, n_snv, 2)  [0]=mut  [1]=non-mut
    genotypes: np.ndarray, # (n_snv, n_clones)
    label,
    output_dir: str,
    threshold: float=0.45
) -> dict:

    counts    = counts.numpy() if hasattr(counts, "numpy") else np.asarray(counts)
    acc_dict  = {}
    clone_ids = sorted(annotation["clone"].unique())

    for clone_id in clone_ids:

        cell_mask    = annotation["clone_hat"] == clone_id
        if not cell_mask.any():
            acc_dict[f"clone_{clone_id}"] = 0.0
            continue
        cell_indices = np.where(cell_mask)[0]
        clone_counts = counts[cell_indices]


        summed = clone_counts.sum(axis=0)
        depth  = summed.sum(axis=-1)
        safe_depth = np.where(depth == 0, 1, depth)
        vaf    = np.where(
                    depth == 0,
                    0.0,
                    summed[..., 0] / safe_depth,
                )

        consensus_genotype = (vaf > threshold).astype(np.int8)
        gt_genotype = genotypes[..., clone_id - 1]

        acc_dict[f"clone_{clone_id}"] = float((consensus_genotype == gt_genotype).mean())


    acc_dict["avg_acc"] = float(np.mean(list(acc_dict.values())))

    json_path = os.path.join(output_dir, f"{label}_genotype_acc.json")
    with open(json_path, 'w') as f:
        json.dump(acc_dict, f, indent=4)


def evaluate_model(model: FACTMx_model|str,
                   sample:str,
                   data: tuple|None=None,
                   annotation: pd.DataFrame|None=None,
                   model_label:str="",
                   save_figs:str='',
                   simulated: bool=False,
                   mut_threshold: float=0.9,
                   ):
    label = f"{sample}_{model_label}"
    data, bcr_df = data if data is not None else prep_data(f'data/{sample}', simulated=simulated)
    annotation = annotation if annotation is not None else pd.read_csv(f'annotation/{sample}_clustering.csv', index_col=0)

    if isinstance(model, str):
        model = FACTMx_model.load(model)

    latent = model.get_latent_representation(data['dataset']).numpy()

    (snv, counts), (bcr, ones) = data['dataset']
    dist_snv, dist_bcr = [head.make_decoder(latent, counts) for head in model.heads]

    os.makedirs(save_figs, exist_ok=True)

    # BCR
    bcr_correct = tf.equal(tf.argmax(bcr, axis=-1), tf.argmax(dist_bcr.probs_parameter(), axis=-1))
    bcr_correct = tf.cast(bcr_correct, tf.float32)

    acc_per_position = np.mean(bcr_correct, axis=0)
    acc_per_cell = np.mean(bcr_correct, axis=-1)

    viz.plot_bcr_acc(acc_per_position, acc_per_cell, heavy_chains.get(sample, 150), label, save_figs)

    # SNV
    re_snv = dist_snv.probs_parameter() * counts[..., None]
    diff = np.abs(snv - re_snv)
    diff_snv = diff.mean(axis=0)
    smape, smape_cell, smape_snv = _smape_error(snv, re_snv)

    viz.plot_snv_rec(diff, diff_snv, smape, smape_snv, label, save_figs)

    # CLONE
    if simulated:
      annotation = _map_prediction2ref(annotation, label=f"{model_label}_jaccard_clone", ref_col="clone", inplace=False)
      viz.plot_latent(latent, bcr_df.clone, 'Clone', label, save_figs)
      viz.cell2clone_acc(annotation, save_figs, label, true_col='clone', pred_col='clone_hat')
      clones_genotypes = np.load(os.path.join(f"data/{sample}", 'clone.npy'))
      _avg_genotype_accuracy(annotation,
                            counts=snv,
                            genotypes=clones_genotypes,
                            label=label,
                            output_dir=save_figs,
                            threshold=mut_threshold)
      viz.snv_genotype_accuracy(annotation,
                            counts=snv,
                            genotypes=clones_genotypes,
                            label=label,
                            output_dir=save_figs,
                            threshold=mut_threshold)