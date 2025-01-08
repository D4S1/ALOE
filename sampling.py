import tensorflow as tf
import tensorflow_probability as tfp


def create_hiperclusters(alpha: float, num_points: int) -> list:
    """
    Generates hyperclusters based on the Chinese Restaurant Process.

    :param alpha: Concentration parameter for the process, controlling the probability of forming new clusters.
    :param num_points: Number of data points to generate clusters for.
    :return: List of cluster sizes.
    """
    hiperclusters = [1]

    for point in range(2, num_points + 1):
        probs = [hc / (point - 1 + alpha) for hc in hiperclusters]
        probs.append(alpha / (point - 1 + alpha))

        hc_idx = tf.random.categorical([probs], 1)  # logits need to be matrix, normally shape is (batch size, categories)

        if hc_idx < len(hiperclusters):
            hiperclusters[int(hc_idx)] += 1
        else:
            hiperclusters.append(1)

    return hiperclusters


def sample_hipercluster_bcr(gene_len: int, n_cells: int, BCR_prior: list) -> tf.Tensor:
    """
    Generates BCR sequences for a hypercluster using categorical distribution with Dirichlet-distributed logits.

    :param gene_len: Length of the BCR gene sequence.
    :param n_cells: Number of cells in the hypercluster.
    :param BCR_prior: Prior probabilities for the Dirichlet distribution (must sum to 1).
    :return: Tensor of shape (n_cells, gene_len, 4) representing one-hot encoded BCR sequences.
    """
    brc_profile_generator = tfp.distributions.Dirichlet(BCR_prior)
    brc_profile = brc_profile_generator.sample(gene_len)
    sample_bcrs = tfp.distributions.OneHotCategorical(brc_profile).sample(n_cells)

    return sample_bcrs


def sample_reads_matrix(n_snp: int, n_cells: int, avg_reads_num: int) -> tf.Tensor:
    """
    Samples a reads matrix from a Poisson distribution.

    :param n_snp: Number of SNP locations.
    :param n_cells: Number of cells to generate reads for.
    :param avg_reads_num: Average number of reads per SNP location.
    :return: Tensor of shape (n_snp, n_cells) representing the reads matrix.
    """
    return tf.random.poisson([n_snp, n_cells], lam=avg_reads_num)


def sample_clone_profile(n_snp: int, n_clones: int, mutation_rate: float, clone_relax_rate: float) -> tf.Tensor:
    """
    Samples the clone mutation profile using Binomial distributions.

    :param n_snp: Number of SNP locations.
    :param n_clones: Number of clones.
    :param mutation_rate: Probability of mutation at each SNP location.
    :param clone_relax_rate: Relaxation rate for mutations, adjusting probabilities.
    :return: Tensor of shape (n_snp, n_clones) representing mutation profiles for each clone.
    """
    omega = tfp.distributions.Binomial(1, mutation_rate).sample((n_snp, n_clones))
    probs = abs(omega - clone_relax_rate)

    return tfp.distributions.Binomial(1, probs).sample()


def sample_data(
    BCR_len: int = 300,
    n_cells: int = 1000,
    n_clones: int = 3,
    n_mutation: int = 100,
    BCR_prior: list = [0.1, 0.1, 0.1, 0.1],
    alpha: float = 40,
    clone_relax_prior: list = [1, 9],
    theta0_prior: list = [0.2, 99.8],
    theta1_prior: list = [45, 55],
    mutation_rate: float = 0.3,
    avg_reads_num: int = 1,
) -> tuple:
    """
    Samples synthetic data including BCR sequences and clone mutation profiles.

    :param BCR_len: Length of the BCR gene sequences.
    :param n_cells: Total number of cells.
    :param n_clones: Number of distinct clones.
    :param n_mutation: Number of SNP locations to analyze.
    :param BCR_prior: Prior probabilities for the Dirichlet distribution.
    :param alpha: Concentration parameter for hypercluster generation.
    :param clone_relax_prior: Prior for the Beta distribution to sample clone relaxation rates.
    :param theta0_prior: Prior for the Beta distribution for theta0.
    :param theta1_prior: Prior for the Beta distribution for theta1.
    :param mutation_rate: Probability of mutation at each SNP location.
    :param avg_reads_num: Average number of reads per SNP location.
    :return: Tuple containing:
             - BCR: Tensor of shape (n_cells, BCR_len, 4), one-hot encoded BCR sequences.
             - clones_profiles: Tensor of shape (n_snp, n_clones), mutation profiles for clones.
    """
    hiperclusters = create_hiperclusters(alpha=alpha, num_points=n_cells)

    n_hc = len(hiperclusters)
    psi = [1 / n_clones] * n_clones
    hc_clone = tf.random.categorical([psi], n_hc)

    BCR = [sample_hipercluster_bcr(gene_len=BCR_len, n_cells=hc, BCR_prior=BCR_prior) for hc in hiperclusters]
    BCR = tf.concat(BCR, axis=0)

    clone_relax_rate = tfp.distributions.Beta(*clone_relax_prior).sample()
    clones_profiles = sample_clone_profile(n_snp=n_mutation, n_clones=n_clones, mutation_rate=mutation_rate, clone_relax_rate=clone_relax_rate)

    # reads sampling
    # mutation sampling

    return BCR, clones_profiles



BCR, clones_profiles = sample_data(n_cells = 100)
print(f'{BCR.shape=}\t{clones_profiles.shape=}')