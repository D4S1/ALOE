# class FACTMx_head_Multinomial(FACTMx_head) -> dim : liczba kategori - 1 (scRNA -> 1) BCR-> 3
import tensorflow as tf
from tensorflow import keras
from FACTMx.FACTMx_head import FACTMx_head_Multinomial
import math


class Encoder(tf.Module):

    def __init__(self, input_dim, latent_dim):
        pass


class Net(tf.Module):

    def __init__(self,
                 input_dim: dict, # {'SNP': (# cells, # SNP, 2), 'BCR': (# cells, # BCR length, 4)} don't pass cells - they are presentig batch
                 latent_dim: int
                ):
        
        self.snp_head = FACTMx_head_Multinomial(dim=1, dim_latent=latent_dim, head_name="SNP")
        self.bcr_head = FACTMx_head_Multinomial(dim=3, dim_latent=latent_dim, head_name="BCR")

        self.flat = keras.layers.Flatten()

        encoder_input_size = math.prod(input_dim['SNP']) + math.prod(input_dim['BCR']) # to be corrected remove batch
        self.encoder = Encoder(encoder_input_size, latent_dim) # to be implemented


    def encode(self, x): # to be change
        snp_encoded = self.snp_head.encode(x)
        bcr_encoded = self.bcr_head.encode(x)

        encoder_input = tf.concat(self.flat(snp_encoded), bcr_encoded, axis=0)

        mu, log_var = self.encoder(encoder_input)

        return mu, log_var

    def deccode(self, z):
        snp_decoded = self.snp_head.decode(z)
        bcr_decoded = self.bcr_head.decode(z)

        return snp_decoded, bcr_decoded
    
    def _reparametrization(self, mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(log_var.shape)
        z = mu + eps*tf.exp(log_var/2)
        return z
    
    def call(self, x):

        mu, log_var = self.encode(x)
        z = self._reparametrization(mu, log_var)
        snp_decoded, bcr_decoded = self.deccode(z)

        return {
            'snp': snp_decoded,
            'bcr': bcr_decoded,
            'mu': mu,
            'log_var': log_var
        }
    
    # loss to be implemented