import tensorflow as tf
import tensorflow_probability as tfp

def sample_alpha(n_dim:int=4, min_large:float=0.85) -> tf.Tensor:
    """
    Given number of dimension of Dirichlet distribution randomly select
    probability of clusters, with one of the cluster beeing significantly more
    probably (parameter min_large)

    :param n_dim: number categories
    :param min_large: what is minimal probabilty for most probable category
    :return: tensor containing probability of each category
    """
    # Ensure valid input
    if n_dim < 2:
        raise ValueError("n must be at least 2 to allow one large number and others.")
    
    # Define the probability for most probable nucleotide
    large_number = min_large + tf.random.uniform(shape=[], minval=0, maxval=1 - min_large)
    
    # Probability of rest of nucleotides
    remaining_sum = 1 - large_number
    
    # Sample (n_dim - 1) numbers from Dirichlet distribution
    alpha = tf.ones([n_dim-1])

    dirichlet_samples = tf.random.gamma(shape=[1], alpha=alpha) 
    dirichlet_samples = dirichlet_samples / tf.reduce_sum(dirichlet_samples)
    remaining_numbers = dirichlet_samples[0] * remaining_sum  # Scale to the remaining sum
    
    # Combine the large number with the rest
    result = tf.concat([remaining_numbers, [large_number]], axis=0)
    result = tf.random.shuffle(result)  # Shuffle to randomize the position of the large number

    return result

def sample_bcr_clone(gene_len:int, n_cells:int, alpha:tf.Tensor) -> tf.Tensor:
    """
    Given gene length and number of cells generate sample gene sequence sampling
    from Dirichlet distribution with given nucleotide profile (alpha) and encode
    result sequence

    :param gene_len: The length of gene that is generated
    :param n_cells: number of cells that have similar bcr and represent same clone
    :param alpha: Tensor of shape (number of categories,) for genes (4,) cause there 4 nucleotides "ATGC"
    :results: Tensor of shape (gene_len, n_cells, 4) that is one hot encoding representation of generated gene
    """

    # Initialize clone nucleotide profile
    clone = tfp.distributions.Dirichlet(alpha)

    # Sample the gene sequence
    sample_bcrs = tf.argmax(clone.sample([gene_len, n_cells]), axis=-1) # get sampled nucleotides per position in gene
    sample_bcrs_encoded = tf.one_hot(sample_bcrs, len(alpha)) # one hot encode nucleotides

    return sample_bcrs_encoded

def sample_reads_matrix(n_snp:int, n_cells:int, avg_reads_num:int) -> tf.Tensor:
    """
    Given number of snp locations and cells with addtitionall information
    about average number of reads per snp location it resturn sampled from poisson
    distribution reads matrix.
    
    :param n_snp: Number of analyzed snp locations
    :param n_cells: Number of cells to generate reads for
    "return: Reads matrix - tesor of shape (n_snp, n_cells)
    """
    return tf.random.poisson([n_snp, n_cells], lam=avg_reads_num)

def sample_mutation_matrix(n_snp:int, n_cells:int, mutation_rate:float) -> tf.Tensor:
    pass
    