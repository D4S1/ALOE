import os
import pandas as pd
import tensorflow as tf
from typing import List

from FACTMx.FACTMx_head import FACTMx_head_Multinomial
from FACTMx.FACTMx_encoder import FACTMx_encoder_Linear
from FACTMx.FACTMx_model import FACTMx_model

from utils.preprocessing import prep_data
from utils.visualization import plot_losses, plot_latent
from utils.architecture import create_config
from utils.clustering import cluster_cells


def train_model(data:dict,
                latent: int,
                layer_configs:dict,
                hypers: dict,
                dims: dict=None,
                loss_scales: List[float]=[1.0, 1.0, 1.0]
                ) -> FACTMx_model:
    snv_dim = dims['snv'] if layer_configs['snv'].get('preencoder', False) else data['n_snv']*2
    bcr_dim = dims['bcr'] if layer_configs['bcr'].get('preencoder', False) else data['n_bcr']*4

    snv_head = FACTMx_head_Multinomial(dim_pos=data['n_snv'],
                                       dim_cat=2,
                                       dim=snv_dim,
                                       dim_latent=latent, head_name="SNV",
                                       layer_configs=layer_configs['snv']
                                       ).get_config()
    bcr_head = FACTMx_head_Multinomial(dim_pos=data['n_bcr'],
                                       dim_cat=4,
                                       dim=bcr_dim,
                                       dim_latent=latent, head_name="BCR",
                                       layer_configs=layer_configs['bcr']
                                       ).get_config()
    encoder = FACTMx_encoder_Linear(latent,
                                    [snv_dim, bcr_dim],
                                    layer_configs=layer_configs['encoder']
                                    ).get_config()
    model = FACTMx_model(latent,
                         heads_config=[snv_head, bcr_head],
                         encoder_config=encoder,
                         loss_scales=loss_scales
                         )

    model.optimizer = tf.keras.optimizers.Adam(learning_rate=hypers['lr'])
    losses = model.train(tf.data.Dataset.from_tensor_slices(data['train']), # validation dataset return train_loss, val_loss
                validation_dataset=data['val'],
                epochs=hypers['epochs'],
                batch_size=hypers['batch_size'],
                shuffle=True,
                )
    return model, losses


def test_models(
    data_path: str,
    hypers:dict,
    models_versions: list,
    output_dir: str="",
    simulated: bool=False
    ):
    sample_name = os.path.basename(data_path)
    output_dir = output_dir if output_dir else "output"
    fig_path = os.path.join(output_dir, 'figs/')
    model_path = os.path.join(output_dir, 'models/')
    annotation_path = os.path.join(output_dir, 'annotations/')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(annotation_path, exist_ok=True)

    print(f"Loading: {sample_name}")
    data, clustering_df = prep_data(data_path, 3, simulated=simulated)
    
    configs = [create_config(data['n_snv'], data['n_bcr'], **m) for m in models_versions]

    for config in configs:
        print(f"Training {config['name']}")
        model_label = f"{sample_name}_{config['name']}"
        model_fig_path = os.path.join(fig_path, model_label)
        os.makedirs(model_fig_path, exist_ok=True)
        model, losses = train_model(data,
                                    latent=config['latent'],
                                    layer_configs=config['layer_configs'],
                                    dims=config['dims'],
                                    hypers=hypers,
                                    loss_scales=config['loss_scales']
                                    )
        plot_losses(losses[0], losses[1], save_path=os.path.join(model_fig_path, f"{model_label}_losses.png"))

        latent = model.get_latent_representation(data['dataset']).numpy()
        label_df = cluster_cells(
            latent=latent,
            counts=data['dataset'][0][0],
            label=config['name'],
        )
        label_df.index = clustering_df.index
        clustering_df = pd.concat([clustering_df, label_df], axis=1)

        model.save(os.path.join(model_path, model_label))

    clustering_df.to_csv(os.path.join(annotation_path, f'{sample_name}_clustering.csv'))