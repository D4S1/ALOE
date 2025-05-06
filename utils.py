import os
import pandas as pd
import tensorflow as tf

import sampling


class GeneratedDataset:

    def __init__(self, config_dict):

        # BCR related data
        self.bcr = config_dict['BCR']
        self.bcr_profiles = config_dict['BCR_profiles']

        self.theta0 = config_dict['theta0']
        self.theta1 = config_dict['theta1']

        # SNP related data
        self.snp = tf.concat([config_dict['reads_counts'], config_dict['mutation_counts']], 0)
        self.clone_profiles = config_dict['clone_profiles']

        self.omega = config_dict['omega']
        self.relax_rate = config_dict['relax_rate'] 

        # Labels
        #self.annotation = tf.convert_to_tensor(config_dict['metadata'])
        self.hc_clone = config_dict['hc_clone']

        self.dataset = self._create_dataset()


    def _create_dataset(self):

        return tf.data.Dataset.from_tensor_slices((self.snp, self.bcr))

    def get_dataset(self):
        return self.dataset

config_dict = sampling.sample_data(n_cells=200)

training_data = GeneratedDataset(config_dict)
tf_dataset = training_data.get_dataset()

print(tf_dataset)

# Shuffle and batch the dataset similar to PyTorch DataLoader
# Adjust buffer_size based on the size of your dataset for optimal shuffling
train_dataset = tf_dataset.shuffle(buffer_size=1000).batch(50)

# # Now you can iterate over the train_dataset in your training loop
# for batch_images, batch_labels in train_dataset.take(1):
#     print(batch_images.shape, batch_labels)
