from utils.model import test_models
from pathlib import Path
import gc
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.preprocessing import prep_data
from utils.evaluate import evaluate_model
from utils.report import create_model_report


models_versions = [
    {'hid_dim_snv': 60, 'hid_dim_bcr': 60, 'latent_dim': 10, 'label': "V2_10_norm", 'linear': True},
    {'hid_dim_snv': 80, 'hid_dim_bcr': 80, 'latent_dim': 20, 'label': "V2_20_norm", 'linear': True},
    {'hid_dim_snv': 100, 'hid_dim_bcr': 100, 'latent_dim': 30, 'label': "V2_30_norm", 'linear': True},
    {'hid_dim_snv': 60, 'hid_dim_bcr': 60, 'latent_dim': 10, 'label': "V3_10_norm"},
    {'hid_dim_snv': 80, 'hid_dim_bcr': 80, 'latent_dim': 20, 'label': "V3_20_norm"},
    {'hid_dim_snv': 100, 'hid_dim_bcr': 100, 'latent_dim': 30, 'label': "V3_30_norm"},
]

samples_names = [
    # "reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "reference_6", "reference_7", "reference_8", "reference_9", "reference_10",
    # "high_1", "high_2", "high_3", "high_4", "high_5", "high_6", "high_7", "high_8", "high_9", "high_10",
    # "low_1", "low_2", "low_3", "low_4", "low_5", "low_6", "low_7", "low_8", "low_9", "low_10",
    # "sparse_1", "sparse_2", "sparse_3", "sparse_4", "sparse_5", "sparse_6", "sparse_7", "sparse_8", "sparse_9", "sparse_10", # dużo hyperclusters
    #"variance_1", "variance_2", "variance_3", "variance_4", "variance_5", "variance_6", "variance_7", "variance_8", "variance_9", "variance_10", # hypercluster rozmyty sekwencja
    # "hh_1", "hh_2", "hh_3", "hh_4", "hh_5", "hh_6", "hh_7", "hh_8", "hh_9", "hh_10", # 80 (ile huperclustrów ma te zachowania), 80 (ile procent komórek z klastra ma sekwencje centroidu)
    # "hl_1", "hl_2", "hl_3", "hl_4", "hl_5", "hl_6", "hl_7", "hl_8", "hl_9", "hl_10", # 80 20
    # "lh_1", "lh_2", "lh_3", "lh_4", "lh_5", "lh_6", "lh_7", "lh_8", "lh_9", "lh_10", # 20 80
    # "ll_1", "ll_2", "ll_3", "ll_4", "ll_5", "ll_6", "ll_7", "ll_8",
    "ll_9", "ll_10",     # 20 20
]


run_output = Path("/home/hoshi/ME/Master/runs/run_09_06_e200")



for sample in samples_names:
  print(f"Evaluating sample {sample}...")
  simulated = 'K' not in sample
  data = prep_data(f'data/sim2/{sample}', simulated=simulated)
  for model in models_versions:
    
    evaluate_model(
        model= run_output / f"models/{sample}_{model['label']}",
        sample=sample,
        data = data,
        data_path = Path(f"data/sim2/{sample}"),
        model_label=model['label'],
        input_dir= run_output,
        simulated=simulated
        )
    tf.keras.backend.clear_session()
    plt.close('all')
    gc.collect()

  del data
  gc.collect()
