import tensorflow as tf
import tensorflow_probability as tfp


def create_hiperclusters(alpha: float, num_points: int):
    hiperclusters = [1]

    for point in range(2, num_points + 1):

        probs = [hc/(point - 1 + alpha) for hc in hiperclusters]
        probs.append(alpha/(point - 1 + alpha))

        hc_idx = tf.random.categorical([probs], 1) # logits need to be matrix, normally shape is (batch size, categories)

        if hc_idx < len(hiperclusters):
            hiperclusters[int(hc_idx)] += 1
        else:
            hiperclusters.append(1)

    return hiperclusters


def sample_hipercluster_bcr(gene_len:int, n_cells:int, BCR_prior:list) -> tf.Tensor:
    """
    Given gene length and number of cells, generates sample bcr sequences for hipercluster
    from categorical distribution with logits sampled from dirichlet distribution
    and one-hot encoded result sequence
    
    :param gene_len: The length of gene that is generated
    :param n_cells: number of cells that have similar bcr and represent same clone
    :results: Tensor of shape (gene_len, n_cells, 4) that is one hot encoding representation of generated gene
    """

    # Initialize clone nucleotide profile
    brc_profile_generator = tfp.distributions.Dirichlet(BCR_prior)

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

def sample_clone_profile(n_snp:int, n_cells:int, mutation_rate:float, clone_relax_rate: float) -> tf.Tensor:

    omega = tfp.distributions.Binomial(1, mutation_rate).sample((n_snp, n_cells))
    probs = abs(omega - clone_relax_rate)

    return tfp.distributions.Binomial(1, probs).sample()


def sample_data(
        BCR_len: int = 300,
        n_cells: int = 1000,
        n_clones: int = 3,
        n_mutation: int = 100,
        BCR_prior: list = [0.1, 0.1, 0.1, 0.1],
        alpha:float = 40,
        clone_relax_prior: list = [1, 9],
        theta0_prior: list = [0.2, 99.8],
        theta1_prior: list = [45, 55],
        mutation_rate: float = 0.3,
        avg_reads_num: int = 1,

):
    # Create hiperclusters
    hiperclusters = create_hiperclusters(alpha=alpha, num_points=n_cells)

    # Make hipercluster - clone assignment
    n_hc = len(hiperclusters)
    psi = [1/n_clones] * n_clones
    hc_clone = tf.random.categorical([psi], n_hc)

    # Sample BCRs
    BCR = tf.zeros([n_cells, BCR_len, 4])
    i = 0
    for hc in hiperclusters:
        # tu jest błąd przypisywanie na slice nie działa
        BCR[i: i+hc,:,:] = sample_hipercluster_bcr(gene_len=BCR_len, n_cells=n_cells, BCR_prior=BCR_prior)
        i += hc
    
    # Sample clone mutation profile
    clones_profiles = tf.zeros((n_clones, n_mutation, n_cells))
    clone_relax_rate = tfp.distributions.Beta(*clone_relax_prior).sample()
    for c in range(n_clones):
        # same here
        clones_profiles[c,:,:] = sample_clone_profile(n_snp=n_mutation, n_cells=n_cells, mutation_rate=mutation_rate, clone_relax_rate=clone_relax_rate)

    return BCR, clones_profiles


print(sample_data(n_cells = 100))