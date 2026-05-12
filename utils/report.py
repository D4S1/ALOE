import os
import json
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape

REPORT_TEMPLATE_PATH  = "templates/report_template_v4.html"   # path to uploaded template

# ---------------------------------------------------------------------------
# Architecture decoder
# ---------------------------------------------------------------------------

def _parse_architecture(label: str) -> dict:
    """
    Decode VAE architecture from a label of the form:
        {sample_id}_V{1|2|3}_{latent_size}

    The last two underscore-separated tokens are always version and latent size;
    everything before them is the sample_id (may itself contain underscores).

    Returns a dict consumed directly by the Jinja2 template.
    """
    normalized = "_norm" in label
    label = label.replace("_norm", "")

    parts = label.split("_")
    version_token = parts[-2]   # e.g. "V1", "V2", "V3"
    latent_token  = parts[-1]   # e.g. "32", "64"

    # Normalise version
    version = version_token.upper()
    if version not in ("V1", "V2", "V3"):
        version = "unknown"

    # Latent size
    try:
        latent_size = int(latent_token)
    except ValueError:
        latent_size = latent_token  # keep raw string if non-numeric

    # Architecture descriptions per version
    descriptions = {
        "V1": {
            "pre_encoder_bcr": "Flatten",
            "pre_encoder_snv": "Flatten",
            "encoder":         "Linear layer → μ / log σ²",
            "decoder_bcr":     "Linear",
            "decoder_snv":     "Linear",
        },
        "V2": {
            "pre_encoder_bcr": "MLP",
            "pre_encoder_snv": "MLP",
            "encoder":         "Linear layer → μ / log σ²",
            "decoder_bcr":     "Linear",
            "decoder_snv":     "Linear",
        },
        "V3": {
            "pre_encoder_bcr": "MLP",
            "pre_encoder_snv": "MLP",
            "encoder":         "MLP → μ / log σ²",
            "decoder_bcr":     "MLP",
            "decoder_snv":     "MLP",
        },
    }

    arch = descriptions.get(version, {
        "pre_encoder_bcr": "—",
        "pre_encoder_snv": "—",
        "encoder":         "—",
        "decoder_bcr":     "—",
        "decoder_snv":     "—",
    })

    return {
        "version":          version,
        "latent_size":      latent_size,
        "layer_norm":       normalized,
        "pre_encoder_bcr":  arch["pre_encoder_bcr"],
        "pre_encoder_snv":  arch["pre_encoder_snv"],
        "encoder":          arch["encoder"],
        "decoder_bcr":      arch["decoder_bcr"],
        "decoder_snv":      arch["decoder_snv"],
    }


def create_model_report(
    label: str,
    input_dir: str,
    output_html: str | None = None,
    extra_notes: str = "",
    simulated: bool = False,
    template_path: str = REPORT_TEMPLATE_PATH,
    clone_acc_thresholds: tuple[float, float] = (0.50, 0.80),
):
    """
    Generate a self-contained HTML model report.
 
    Parameters
    ----------
    label                : Run/sample identifier, format: {id}_V{1|2|3}_{latent}.
    fig_path             : Directory that contains all figure files.
    output_html          : Destination path for the rendered report.
                           Defaults to <fig_path>/../reports/{label}_report.html.
    extra_notes          : Free-text appended to the Notes section.
    simulated            : When True, the Clone Analysis section is shown.
    template_path        : Path to report_template_v3.html (upload to Colab root).
    clone_acc_thresholds : (yellow_cutoff, green_cutoff) tuple.
                           < yellow → red, < green → yellow, >= green → green.
                           Default: (0.50, 0.80)
    """
 
    # ── Output paths ────────────────────────────────────────────────────────
    report_dir = os.path.join(input_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
 
    if output_html is None:
        output_html = os.path.join(report_dir, f"{label}_report.html")

    fig_path = input_dir / f'figs/{label}'
    rel_prefix = os.path.relpath(fig_path, report_dir)
 
    # ── Architecture from label ──────────────────────────────────────────────
    architecture = _parse_architecture(label)
 
    # ── Metrics / Accuracy ───────────────────────────────────────────────────
    json_path = fig_path / 'metrics.json'

    clone_acc = None
    genotype_acc = None
    hypercluster_ari = None

    if json_path.exists():
        with open(json_path) as f:
            metrics = json.load(f)
            clone_acc = metrics.get('clone')
            genotype_acc = metrics.get('genotype')
            hypercluster_ari = metrics.get('hypercluster').get('ari')
 
    # ── Genotype plot file detection ─────────────────────────────────────────
    genotype_html_exists = simulated and os.path.exists(
        os.path.join(fig_path, f"{label}_genotype_acc.html")
    )
 
    # ── SNV accuracy Plotly figure (available for all samples) ───────────────
    snv_accuracy_exists = os.path.exists(
        os.path.join(fig_path, f"{label}_snv_accuracy.html")
    )
 
    # ── Jinja2 environment ───────────────────────────────────────────────────
    template_dir  = os.path.dirname(os.path.abspath(template_path))
    template_file = os.path.basename(template_path)
 
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(disabled_extensions=("html",)),
    )
 
    # Expose fig_path helper as a callable inside the template
    def fig_path_fn(fname):
        return os.path.join(rel_prefix, fname)
 
    env.globals["fig_path"] = fig_path_fn
 
    template = env.get_template(template_file)
 
    # ── Render ───────────────────────────────────────────────────────────────
    html = template.render(
        label                 = label,
        generated_at          = datetime.now().strftime("%Y-%m-%d %H:%M"),
        architecture          = architecture,
        simulated             = simulated,
        clone_acc             = clone_acc,
        genotype_acc          = genotype_acc,
        clone_acc_thresholds  = clone_acc_thresholds,
        hypercluster_ari      = hypercluster_ari,
        genotype_html_exists  = genotype_html_exists,
        snv_accuracy_exists   = snv_accuracy_exists,
        notes                 = extra_notes if extra_notes else "No notes provided.",
    )
 
    with open(output_html, "w") as f:
        f.write(html)
 
    print("Saved report:", output_html)
