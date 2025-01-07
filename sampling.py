import tensorflow as tf
import tensorflow_probability as tfp

def sample_bcr_hipercluster(gene_len:int, n_cells:int) -> tf.Tensor:
    """
    Given gene length and number of cells, generates sample bcr sequences for hipercluster
    from categorical distribution with logits sampled from dirichlet distribution
    and one-hot encoded result sequence
    
    :param gene_len: The length of gene that is generated
    :param n_cells: number of cells that have similar bcr and represent same clone
    :results: Tensor of shape (gene_len, n_cells, 4) that is one hot encoding representation of generated gene
    """

    # Initialize clone nucleotide profile
    brc_profile_generator = tfp.distributions.Dirichlet([0.1, 0.1, 0.1, 0.1])

    # for each cell get brc profile by sampling (gen len,  

    # Sample the gene sequence
    brc_profile = brc_profile_generator.sample([gene_len])
    sample_bcrs = tf.random.categorical(brc_profile, n_cells)
    sample_bcrs_encoded = tf.one_hot(sample_bcrs, 4) # one hot encode nucleotides

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

def sample_clone_mutation(n_snp:int, n_cells:int, variant_rate:float) -> tf.Tensor:
    # omega = matrix with 1s and 0s symbolizing presents of ith mutation
    # err = sample from beta distribution with alpha and beta taken from k
    # hyperparam
    # C = matrix with real mutation profile based on omega matrix with error rate err
    # return C
    pass
    

def sample_clone():
    # inuput params: bcr_len, n_cells, n_snp,
    
    # sample alpha - clone profile
    # bcr sample
    # reads matrix
    # mutations
    pass