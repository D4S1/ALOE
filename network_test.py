# class FACTMx_head_Multinomial(FACTMx_head) -> dim : liczba kategori - 1 (scRNA -> 1) BCR-> 3
import tensorflow as tf
from FACTMx.FACTMx_head import FACTMx_head_Multinomial


class Encoder(tf.Module):

    def __init__(self, input_dim, encoder_dim_latent):
        pass


class Net(tf.Module):

    def __init__(self,
                 input_dim: dict, # {'mutation': (# cells, # SNP, 2), 'BCR': (# cells, # BCR length, 4)}
                 encoder_dim_latent,
                 decoder_input_dim
                ):
        
        self.mutation_head = FACTMx_head_Multinomial(dim=1, dim_latent=decoder_input_dim, head_name="Mutations")
        self.bcr_head = FACTMx_head_Multinomial(dim=3, dim_latent=decoder_input_dim, head_name="BCR")

        # input size -> encoder input
        self.merge_heads = None

        # encoder input -> main latent
        self.encoder = Encoder(None, encoder_dim_latent)

        # sample from latent -> split to heads (..., head_dim=2)
        self.split_heads = None

    def encode(self, data):
        mutation_encoded = self.mutation_head(data)
        bcr_encoded = self.bcr_head(data)

        # mutation -> linear, Relu ,linear
        # bcr -> conv -> flaten, linear

        # stack tensors (cells, hid_dim, heads)
        # 
        # encoder(stacked tensors)        

    def decode(self):
        pass