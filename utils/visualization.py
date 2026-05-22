from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import os
import json


sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def plot_losses(train_losses,
                val_losses=None,
                save_path: str = ""
                ):
    # convert tensors → numpy
    train_losses = tf.stack(train_losses).numpy()

    if val_losses is not None:
        val_losses = tf.stack(val_losses).numpy()

    val_scaler = math.ceil(len(train_losses) / len(val_losses))

    # nicer theme

    sns.set_theme(style="whitegrid", font_scale=1.1)

    plt.figure(figsize=(12, 6))   # rectangular

    sns.lineplot(x=np.arange(len(train_losses)), y=train_losses, label="Train Loss", lw=2)

    if val_losses is not None:
        sns.lineplot(x=np.arange(1, len(val_losses)+1)*val_scaler - 1, y=val_losses, label="Validation Loss", lw=2)

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss", fontsize=16, weight="bold")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latent(
        z_red,
        clusters,
        name,
        label="",
        save_path="",
    ):
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame({
        "UMAP1": z_red[:,0],
        "UMAP2": z_red[:,1],
        name: clusters.astype(str)
    })
    df.sort_values(name, inplace=True)

    fig = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        color=name,
        opacity=0.7,
        title=f"Latent Space — {label} ({name})"
    )

    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=30,r=150,t=50,b=30),
        template="plotly_white"
    )


    os.makedirs(save_path, exist_ok=True)

    outfile = os.path.join(save_path, f"{label}_latent_{name.lower().replace(' ', '_')}.html")

    fig.write_html(
        outfile,
        include_plotlyjs="cdn",
        config={"responsive": True}
    )
    del fig


def cell2clone_acc(df: pd.DataFrame,
                  output_dir: str,
                  label: str,
                  true_col: str='clone',
                  pred_col: str='clone_hat'
                  ):
    os.makedirs(output_dir, exist_ok=True)

    labels = sorted(df[true_col].unique())
    conf_matrix = confusion_matrix(df[true_col], df[pred_col], labels=labels)
    acc_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Dict
    acc_dict = {
        "overall": float(accuracy_score(df[true_col], df[pred_col]))
    }
    for i, clone_id in enumerate(labels):
        acc_dict[f"clone_{clone_id}"] = float(acc_per_class[i])

    # json_path = os.path.join(output_dir, f"{label}_clone_acc.json")
    # with open(json_path, 'w') as f:
    #     json.dump(acc_dict, f, indent=4)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {label}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plot_path = os.path.join(output_dir, f"{label}_{pred_col}_clone_acc.png")
    plt.savefig(plot_path)
    plt.close()

    return acc_dict


def plot_bcr_acc(
    acc_per_position,
    acc_per_cell,
    heavy_chain,
    label="",
    save_path=""
    ):

    prefix = f"{label}_" if label else ""
    os.makedirs(save_path, exist_ok=True)

    # Accuracy per position
    fig, ax = plt.subplots(figsize=(10,4))

    x_pos = np.arange(1, len(acc_per_position)+1)

    sns.lineplot(x=x_pos, y=acc_per_position, linewidth=2, ax=ax)

    ax.axvline(
        heavy_chain,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Heavy chain"
    )

    ax.set_title(f"BCR Accuracy per Position {label}")
    ax.set_xlabel("BCR position")
    ax.set_ylabel("Accuracy")

    ax.set_ylim(max(0.0, acc_per_position.min()-0.02), 1.01)

    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(os.path.join(save_path, f"{prefix}bcr_acc_pos.png"))
    plt.close(fig)


    # Accuracy per cell
    fig, ax = plt.subplots(figsize=(10,4))

    sns.histplot(acc_per_cell, bins=50, kde=True, ax=ax)

    ax.set_title(f"BCR Accuracy per Cell {label}")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Cells")

    ax.set_xlim(max(0, acc_per_cell.min()-0.01), 1.0)

    fig.tight_layout()

    fig.savefig(os.path.join(save_path, f"{prefix}bcr_acc_cell.png"))
    plt.close(fig)


def plot_snv_rec(
    diff,
    diff_snv,
    smape,
    smape_snv,
    label="",
    save_path=""
    ):

    os.makedirs(save_path, exist_ok=True)

    prefix = f"{label}_" if label else ""

    plot_configs = [

        (diff[...,0], True, "diff_mut", "Mutation reconstruction error"),
        (diff[...,1], True, "diff_nmut", "Non-mutation reconstruction error"),

        (diff_snv[...,0], False, "diff_snv_mut", "Average mutation error per SNV"),
        (diff_snv[...,1], False, "diff_snv_nmut", "Average non-mutation error per SNV"),

        (smape[...,0], True, "smape_mut", "Mutation SMAPE"),
        (smape[...,1], True, "smape_nmut", "Non-mutation SMAPE"),

        (smape_snv[...,0], False, "smape_snv_mut", "Average mutation SMAPE per SNV"),
        (smape_snv[...,1], False, "smape_snv_nmut", "Average non-mutation SMAPE per SNV"),
    ]


    for data, is_heatmap, fname, title in plot_configs:

        full_title = f"{label} - {title}" if label else title

        if is_heatmap:

            fig, ax = plt.subplots(figsize=(8,5))

            sns.heatmap(
                data,
                cmap="RdBu_r",
                center=0,
                cbar_kws={"label":"Error"},
                ax=ax
            )

            ax.set_xlabel("SNV")
            ax.set_ylabel("Cell")

        else:

            fig, ax = plt.subplots(figsize=(14,3))

            sns.barplot(x=np.arange(data.shape[0]), y=data, ax=ax)

            ax.set_xlabel("SNV")
            ax.set_ylabel("Error")
            plt.xticks(rotation=90, fontsize=8)


        ax.set_title(full_title)

        fig.tight_layout()

        fig.savefig(os.path.join(save_path, f"{prefix}{fname}.png"))
        plt.close(fig)


def snv_genotype_accuracy(
    annotation: pd.DataFrame,
    counts: np.ndarray,    # (n_cells, n_snv, 2)  [0]=mut  [1]=non-mut
    genotypes: np.ndarray, # (n_snv, n_clones)
    label: str,
    output_dir: str,
    pred_col: str,
    threshold: float = 0.25,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    counts    = counts.numpy() if hasattr(counts, "numpy") else np.asarray(counts)
    clone_ids = sorted(annotation["clone"].unique())
    acc_cols  = {}

    for clone_id in clone_ids:

        cell_indices = np.where(annotation[pred_col] == clone_id)[0]
        if not cell_indices.size:
            n_snv = genotypes[..., clone_id - 1].shape[0]
            acc_cols[f"clone_{clone_id}"] = np.zeros(n_snv)
            continue
        clone_counts = counts[cell_indices]

        depth      = clone_counts.sum(axis=-1)
        safe_depth = np.where(depth == 0, 1, depth)
        vaf        = np.where(
                         depth == 0,
                         0.0,
                         clone_counts[..., 0] / safe_depth,
                     )

        consensus_genotype = (vaf > threshold).astype(np.int8)
        gt_genotype        = genotypes[..., clone_id - 1]

        acc_cols[f"clone_{clone_id}"] = (
            (consensus_genotype == gt_genotype).mean(axis=0)
        )

    acc_df = pd.DataFrame(acc_cols)
    acc_df.index.name = "snv"

    # ── bold tick labels for SNVs mutated in any clone ────────────────────
    any_mutated = genotypes.any(axis=1)  # (n_snv,)
    ticklabels  = [f"<b>{i}</b>" if m else str(i)
                   for i, m in enumerate(any_mutated)]

    # ── Plotly grouped bar ────────────────────────────────────────────────
    fig = go.Figure()

    for col_idx, col in enumerate(acc_df.columns):
        clone_id   = clone_ids[col_idx]
        gt_col     = genotypes[..., clone_id - 1]
        customdata = np.where(gt_col == 1, "mutated", "not mutated")

        fig.add_trace(go.Bar(
            name=col,
            x=acc_df.index.astype(str),
            y=acc_df[col],
            customdata=customdata,
            hovertemplate=(
                "SNV: %{x}<br>"
                "Accuracy: %{y:.3f}<br>"
                "GT: %{customdata}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        title=f"Per-SNV genotype accuracy per clone — {label}",
        xaxis=dict(
            title="SNV index",
            tickmode="array",
            tickvals=acc_df.index.astype(str).tolist(),
            ticktext=ticklabels,
            tickangle=90,
        ),
        yaxis=dict(
            title="Accuracy",
            range=[0, 1],
        ),
        legend_title="Clone",
        bargap=0.15,
        bargroupgap=0.05,
        template="plotly_white",
        height=500,
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f"{label}_{pred_col}_snv_accuracy.html"))
    else:
        fig.show()

    del fig