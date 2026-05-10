import tensorflow as tf


def _dense_block(units_list, activation='relu', norm=False, dropout_rate=None):
    layers = []
    for u in units_list:
        layers.append(tf.keras.layers.Dense(u, activation=activation))
        if norm:
            layers.append(tf.keras.layers.LayerNormalization())
    if dropout_rate is not None and dropout_rate > 0.0:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    return layers

# === Preencoders ===
def build_preencoder_snv(hid_dim, n_snv: int, norm=False):
    # Final layer defines dim for snv representation
    units = [124, hid_dim]
    seq = tf.keras.Sequential()
    seq.add(tf.keras.Input(shape=(n_snv * 2,)))
    for layer in _dense_block(units, norm=norm):
        seq.add(layer)
    return seq

def build_preencoder_bcr(hid_dim, n_bcr: int,norm=False):
    # Final layer defines dim for BCR representation
    units = [512, 256, 128, hid_dim]
    seq = tf.keras.Sequential()
    seq.add(tf.keras.Input(shape=(n_bcr * 4,)))
    for layer in _dense_block(units, norm=norm):
        seq.add(layer)
    return seq

# === Encoder ===
def build_encoder(latent, snv_out_dim, bcr_out_dim, norm=False):
    # Encoder input = concatenation of preencoder snv and BCR outputs
    input_dim = snv_out_dim + bcr_out_dim
    units = [latent * 4, latent * 2, latent]
    seq = tf.keras.Sequential()
    seq.add(tf.keras.Input(shape=(input_dim,)))
    for layer in _dense_block(units, norm=norm, dropout_rate=(0.1 if norm else None)):
        seq.add(layer)
    return seq

# === Decoders ===
def build_decoder_snv(latent, n_snv: int, norm=False):
    units = [latent * 3, n_snv]
    seq = tf.keras.Sequential()
    seq.add(tf.keras.Input(shape=(latent,)))
    for layer in _dense_block(units[:-1], norm=norm):
        seq.add(layer)
    seq.add(tf.keras.layers.Dense(units[-1], activation='relu'))
    seq.add(tf.keras.layers.Dense(n_snv * 2, activation='log_softmax'))
    return seq

def build_decoder_bcr(latent, n_bcr: int, norm=False):
    units = [latent * 4, 256, 512, n_bcr]
    seq = tf.keras.Sequential()
    seq.add(tf.keras.Input(shape=(latent,)))
    for layer in _dense_block(units[:-1], norm=norm):
        seq.add(layer)
    seq.add(tf.keras.layers.Dense(units[-1], activation='relu'))
    seq.add(tf.keras.layers.Dense(n_bcr * 4, activation='log_softmax'))
    return seq


def create_config(
    n_snv: int,
    n_bcr: int,
    hid_dim_snv: int,
    hid_dim_bcr: int,
    latent_dim: int,
    label: str = "dummy_model",
    linear: bool = False,
    normalized: bool = True,
    loss_scales: list|None = None,
    ) -> dict:
    snv_preencoder = build_preencoder_snv(hid_dim_snv, n_snv, norm=normalized).get_config()
    bcr_preencoder = build_preencoder_bcr(hid_dim_bcr, n_bcr, norm=normalized).get_config()


    if linear:
      config = {
          'snv': {'logits': 'linear', 'preencoder': snv_preencoder},
          'bcr': {'logits': 'linear', 'preencoder': bcr_preencoder},
          'encoder': {'loc':'linear', 'scale':'linear'}
      }
    else:
      encoder = build_encoder(latent_dim, hid_dim_snv, hid_dim_bcr, norm=normalized).get_config()
      snv_decoder = build_decoder_snv(latent_dim, n_snv, norm=True).get_config()
      bcr_decoder = build_decoder_bcr(latent_dim, n_bcr, norm=True).get_config()

      config = {
          'snv': {'logits': snv_decoder, 'preencoder': snv_preencoder},
          'bcr': {'logits': bcr_decoder, 'preencoder': bcr_preencoder},
          'encoder': {'loc': encoder}
      }
    loss_scales = loss_scales if loss_scales is not None else [1 / latent_dim **(1/2), 5 / (n_snv) **(1/2), 1 / (n_bcr)**(1/2)]

    return {'name': label,
            'latent': latent_dim,
            'layer_configs': config,
            'dims': {'snv': hid_dim_snv,'bcr': hid_dim_bcr},
            'loss_scales': loss_scales
            }