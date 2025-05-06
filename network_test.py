# class FACTMx_head_Multinomial(FACTMx_head) -> dim : liczba kategori - 1 (scRNA -> 1) BCR-> 3
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from FACTMx.FACTMx_head import FACTMx_head, FACTMx_head_Bernoulli
from FACTMx.FACTMx_encoder import FACTMx_encoder
from FACTMx.FACTMx_model import FACTMx_model


class FACTMx_head_Multinomial(FACTMx_head):

    head_type = 'Multinomial'

    def __init__(self,
                 dim_pos,
                 dim_cat,
                dim, dim_latent, head_name,
                layer_configs={'logits':'linear'},
                eps = 1E-3, 
                **kwargs):
        super().__init__(dim, dim_latent, head_name)
        self.eps = eps
        self.dim_pos = dim_pos
        self.dim_cat = dim_cat
        self.layers = {}

        logits_config = layer_configs.pop('logits', 'linear')

        if logits_config == 'linear':
            self.layers['logits'] = tf.keras.Sequential(
                                    [tf.keras.Input(shape=(self.dim_latent,)),
                                    tf.keras.layers.Dense(self.dim)]
                            )
        else:
            self.layers['logits'] = tf.keras.Sequential.from_config(logits_config)

        assert self.layers['logits'].output_shape == (None, self.dim)
        assert self.layers['logits'].input_shape == (None, self.dim_latent)

        self.t_vars = self.layers['logits'].trainable_variables

    def decode_params(self, latent):
        #decode logits from a latent point
        return tf.reshape(self.layers['logits'](latent), shape=(-1, self.dim_pos, self.dim_cat))

    def make_decoder(self, latent, counts):
        #return the decoding distribution given its latent point
        logits = self.decode_params(latent)
        padded_logits = tf.pad(logits,
                            tf.constant([[0, 0], [0, 0], [1, 0]]),
                            'CONSTANT')
        return tfp.distributions.Multinomial(total_count=counts, logits=padded_logits)

    def decode(self, latent, data):
        #decode a sample from latent
        counts = data[1]
        return self.make_decoder(latent, counts).sample()

    def encode(self, data):
        #give logits to encode
        logits = tf.math.log(data[0] + self.eps)
        normalized = logits[:,1:] - tf.reshape(logits[:,0], shape=(-1,1))
        return {'encoder_input': normalized}

    def loss(self, data, latent, beta=1):
        #return -loglikelihood of data given its latent point
        observations, counts = data
        log_prob = self.make_decoder(latent, counts).log_prob(observations)

        loss = -tf.reduce_sum(log_prob) / data.shape[0]
        loss += tf.reduce_sum(self.layers['logits'].losses)

        return loss 

    def get_config(self):
        config = {
            'head_type': self.head_type,
            'dim_pos': self.dim_pos,
            'dim_cat': self.dim_cat,
            'dim': self.dim,
            'dim_latent': self.dim_latent,
            'head_name': self.head_name,
            'layer_configs': {'logits': self.layers['logits'].get_config()}
        }
        return config

    def from_config(config):
        return FACTMx_head_Multinomial(**config)
    

LATENT_DIM = 5
SNP_DIM = (100, 1)
BCR_DIM = (300, 3)


snp_head = FACTMx_head_Multinomial(dim_pos=SNP_DIM[0], dim_cat=SNP_DIM[1], dim=1, dim_latent=LATENT_DIM, head_name="SNP").get_config()
bcr_head = FACTMx_head_Multinomial(dim_pos=BCR_DIM[0], dim_cat=BCR_DIM[1], dim=3, dim_latent=LATENT_DIM, head_name="BCR").get_config()


main_encoder = FACTMx_encoder(LATENT_DIM, [(SNP_DIM[0]*SNP_DIM[1]), (BCR_DIM[0]* BCR_DIM[1])]).get_config()
model = FACTMx_model(LATENT_DIM, [snp_head, bcr_head], main_encoder)


print(model.get_config())