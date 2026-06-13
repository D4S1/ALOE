import glob
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

path = Path("/home/hoshi/ME/Master/runs/")

mapping = {
    'high': 'High reads',
    'low': 'Low reads',
    'reference': 'Basic',
    'sparse': 'Sparse clusters',
    'variance': 'High variance BCR',
    'hh': "80% centroids in 80% clusters",
    'hl': "80% centroids in 20% clusters",
    'lh': "20% centroids in 80% clusters",
    'll': "20% centroids in 20% clusters"
}

def summarize_run(run_directory: str) -> pd.DataFrame:
    summary = []
    for model_path in glob.glob(os.path.join(run_directory, "figs/*")):
        model_label = os.path.basename(model_path)
        sample_type, sample_id, *architecture = model_label.split('_')
        metrics_path = os.path.join(run_directory, f"figs/{model_label}/{model_label}_metrics.json")
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            summary.append({
                'sample_type': sample_type,
                'sample_id': sample_id,
                'architecture': '_'.join(architecture),
                'hypercluster_ari': data['hypercluster']['ari'],
                'clone_cityblock': data['clone']['overall']['cityblock'],
                'clone_jaccard': data['clone']['overall']['jaccard'],
                'genotype_cityblock': data['genotype']['avg_acc']['cityblock'],
                'genotype_jaccard': data['genotype']['avg_acc']['jaccard']
            })
    return pd.DataFrame(summary).sort_values(by=['sample_type'])

def viz_summary(summary_df: pd.DataFrame, outdir: str, hue: str | None = None) -> None:
    metrics = ['hypercluster_ari', 'clone_cityblock', 'clone_jaccard', 'genotype_cityblock', 'genotype_jaccard']
    for label, architecture_df in summary_df.groupby('architecture'):
        for metric in metrics:
            metric_type = " based accuracy" if 'ari' not in metric else ""
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 6))
            ax = sns.boxplot(x='sample_type', y=metric, data=architecture_df, hue=hue)
            sns.despine(left=True)
            plt.title(f'{metric.capitalize().replace("_", " ")} for {label}')
            plt.ylabel(metric.capitalize().replace("_", " ") + metric_type)
            plt.xlabel("")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            if hue and ax.get_legend():
                ax.legend(
                    title=hue.capitalize(),
                    bbox_to_anchor=(1.01, 1),
                    loc='upper left',
                    borderaxespad=0
                )
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'{label}_{metric}.png'))
            plt.close()


def single_run(run: str):
    run_path = path / run                          # fix: use the run parameter
    output_dir = run_path / "summary"
    os.makedirs(output_dir, exist_ok=True)
    summary_df = summarize_run(run_path)           # fix: pass run_path, not global path
    summary_df['sample_type'] = pd.Categorical(
        summary_df['sample_type'].map(mapping),
        categories=list(mapping.values()),
        ordered=True
    )
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    viz_summary(summary_df, output_dir)

def multiple_runs(runs: list, outdir: str):
    dfs = []
    for run in runs:
        df = pd.read_csv(path / f"{run}/summary/metrics_summary.csv")  # fix: / not \
        df['run'] = run                            # tag each row with its source run
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)         # stack vertically
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "merged_metrics_summary.csv"), index=False)
    viz_summary(df, outdir, hue='run')             # color boxes by run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize and visualize benchmark run metrics.")
    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more run names (subdirectories of the base path)"
    )
    parser.add_argument(
        "--outdir", "-o",
        default=".",
        help="Output directory for multi-run plots (default: current directory)"
    )
    args = parser.parse_args()

    if len(args.runs) == 1:
        single_run(args.runs[0])
    else:
        multiple_runs(args.runs, args.outdir)
