import glob
import os 
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


def summarize_run(run_directory: str) -> pd.DataFrame:
    summary = []

    for model_path in glob.glob(os.path.join(run_directory, "figs/*")):
        model_label = os.path.basename(model_path)
        sample_type, sample_id, *architecture = model_label.split('_')
        path = os.path.join(run_directory, f"figs/{model_label}/{model_label}_metrics.json")
        with open(path, 'r') as f:
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

def viz_summary(summary_df: pd.DataFrame, outdir: str) -> None:
    metrics = ['hypercluster_ari', 'clone_cityblock', 'clone_jaccard', 'genotype_cityblock', 'genotype_jaccard']

    for label, architecture_df in summary_df.groupby('architecture'):
        for metric in metrics:
            metric_type = "based accuracy" if 'ari' not in metric else ""

            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='sample_type', y=metric, data=architecture_df, color="#851432")
            sns.despine(left=True)
            plt.title(f'{metric.capitalize().replace("_", " ")} for {label}')
            plt.ylabel(metric.capitalize().replace("_", " ") + metric_type)
            plt.xlabel("")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'{label}_{metric}.png'))
            plt.close()


if __name__ == "__main__":
    path = "/home/hoshi/ME/Master/runs/run_12_05_e20"
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
    output_dir = os.path.join(path, "summary")
    os.makedirs(output_dir, exist_ok=True)

    summary_df = summarize_run(path)
    summary_df['sample_type'] = pd.Categorical(
        summary_df['sample_type'].map(mapping),
        categories=mapping.values(),
        ordered=True
    )
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    viz_summary(summary_df, output_dir)