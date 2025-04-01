# class FACTMx_head_Multinomial(FACTMx_head) -> dim : liczba kategori - 1 (scRNA -> 1) BCR-> 3
import tensorflow as tf
from tensorflow import keras
from FACTMx.FACTMx_head import FACTMx_head_Multinomial
import math


class Encoder(tf.Module):

    def __init__(self, input_dim, latent_dim):
        
        self.layers = keras.Sequnetial([
            keras.layers.Dense(1024, acitvation='relu'),
            keras.layers.Dense(512, acitvation='relu'),
            keras.layers.Dense(128, acitvation='relu'),
            keras.layers.Dense(64, acitvation='relu'),
        ])

        self.mu = keras.layers.Dense(latent_dim, acitvation='relu')
        self.logvar = keras.layers.Dense(latent_dim, acitvation='relu')

    
    def call(self, x):

        x = self.layers(x)

        mu = self.mu(x)
        logvar = self.lagvar(x)

        return mu, logvar
    


class Net(tf.Module):

    def __init__(self,
                 input_dim: dict, # {'SNP': (# SNP, 2), 'BCR': (# BCR length, 4)}
                 latent_dim: int
                ):
        
        self.snp_head = FACTMx_head_Multinomial(dim=1, dim_latent=latent_dim, head_name="SNP")
        self.bcr_head = FACTMx_head_Multinomial(dim=3, dim_latent=latent_dim, head_name="BCR")

        self.flat = keras.layers.Flatten()

        encoder_input_size = math.prod(input_dim['SNP']) + math.prod(input_dim['BCR'])
        self.encoder = Encoder(encoder_input_size, latent_dim)


    def encode(self, x_snp, x_bcr):
        snp_encoded = self.snp_head.encode(x_snp)
        bcr_encoded = self.bcr_head.encode(x_bcr)

        encoder_input = tf.concat(self.flat(snp_encoded), bcr_encoded, axis=0)

        mu, log_var = self.encoder(encoder_input)

        return mu, log_var

    def decode(self, z):
        snp_decoded = self.snp_head.decode(z)
        bcr_decoded = self.bcr_head.decode(z)

        return snp_decoded, bcr_decoded
    
    def _reparametrization(self, mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(log_var.shape)
        z = mu + eps*tf.exp(log_var/2.0)
        return z
    
    def call(self, x):

        mu, log_var = self.encode(x)
        z = self._reparametrization(mu, log_var)
        snp_decoded, bcr_decoded = self.decode(z)

        return {
            'snp': snp_decoded,
            'bcr': bcr_decoded,
            'mu': mu,
            'log_var': log_var
        }
    
    def loss(self, data, outputs):

        mu = outputs['mu']
        log_var = outputs['log_var']
        latent = mu + tf.zeros_like(log_var) * tf.exp(log_var / 2.0)

        snp_loss = self.snp_head.loss(data=data.snp, latent=latent)
        bcr_loss = self.bcr_head.loss(data=data.bcr, latent=latent)
        rec_loss = snp_loss + bcr_loss

        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)

        print(kl_loss)

        return rec_loss + kl_loss