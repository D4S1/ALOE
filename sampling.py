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

        hc_idx = tf.random.categorical([probs], 1)[0, 0]  # Flatten the output

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
    sample_bcrs = tfp.distributions.OneHotCategorical(logits=brc_profile).sample(n_cells)

    return sample_bcrs

def sample_reads_matrix(n_snp: int, n_cells: int, avg_reads_num: int) -> tf.Tensor:
    """
    Samples a reads matrix from a Poisson distribution.

    :param n_snp: Number of SNP locations.
    :param n_cells: Number of cells to generate reads for.
    :param avg_reads_num: Average number of reads per SNP location.
    :return: Tensor of shape (n_snp, n_cells) representing the reads matrix.
    """
    return tf.random.poisson(lam=avg_reads_num, shape=[n_snp, n_cells])

def sample_clone_profile(n_snp: int, n_clones: int, mutation_rate: float, clone_relax_rate: float) -> tf.Tensor:
    """
    Samples the clone mutation profile using Binomial distributions.

    :param n_snp: Number of SNP locations.
    :param n_clones: Number of clones.
    :param mutation_rate: Probability of mutation at each SNP location.
    :param clone_relax_rate: Relaxation rate for mutations, adjusting probabilities.
    :return: Tensor of shape (n_snp, n_clones) representing mutation profiles for each clone.
    """
    omega = tfp.distributions.Binomial(total_count=1, probs=mutation_rate).sample([n_snp, n_clones])
    probs = tf.abs(omega - clone_relax_rate)

    return tfp.distributions.Binomial(total_count=1, probs=probs).sample()

def sample_mutation_matrix(
    reads_counts: tf.Tensor,  # shape (n_snp, n_cells)
    hiperclusters: list,
    hc_clone: tf.Tensor,  # shape (n_hc,)
    clones_profiles: tf.Tensor,  # shape (n_snp, n_clones)
    theta0: tf.Tensor,  # scalar
    theta1: tf.Tensor,  # shape (n_snp, 1)
) -> tf.Tensor:  # shape (n_snp, n_cells)
    """
    Samples the mutation matrix based on clone profiles and hyperclusters.

    :param reads_counts: Tensor of shape (n_snp, n_cells) representing the reads matrix.
    :param hiperclusters: List of sizes of hyperclusters.
    :param hc_clone: Tensor of shape (n_hc,) mapping each hypercluster to a clone index.
    :param clones_profiles: Tensor of shape (n_snp, n_clones), mutation profiles for each clone.
    :param theta0: Scalar Tensor, mutation rate when no mutation occurs.
    :param theta1: Tensor of shape (n_snp, 1), mutation rate when a mutation occurs.
    :return: Tensor of shape (n_snp, n_cells) representing the mutation matrix.
    """
    probs = []
    for size, clone in zip(hiperclusters, hc_clone):
        hc_c = tf.tile(
            tf.expand_dims(clones_profiles[:, clone], axis=1), [1, size]
        )
        hc_c = hc_c * theta1 + (1 - hc_c) * theta0
        probs.append(hc_c)
    probs = tf.concat(probs, axis=1)

    return tfp.distributions.Binomial(total_count=reads_counts, probs=probs).sample()

def sample_data(
    n_cells: int = 1000,
    BCR_len: int = 300,
    n_clones: int = 3,
    n_mutation: int = 100,
    BCR_prior: list = [0.1, 0.1, 0.1, 0.1],
    alpha: float = 40.0,
    clone_relax_prior: list = [1, 9],
    theta0_prior: list = [0.2, 99.8],
    theta1_prior: list = [45, 55],
    mutation_rate: float = 0.3,
    avg_reads_num: int = 1,
) -> dict:
    """
    Samples synthetic data including BCR sequences and clone mutation profiles.

    :param n_cells: Total number of cells.
    :param BCR_len: Length of the BCR gene sequences.
    :param n_clones: Number of distinct clones.
    :param n_mutation: Number of SNP locations to analyze.
    :param BCR_prior: Prior probabilities for the Dirichlet distribution.
    :param alpha: Concentration parameter for hypercluster generation.
    :param clone_relax_prior: Prior for the Beta distribution to sample clone relaxation rates.
    :param theta0_prior: Prior for the Beta distribution for theta0.
    :param theta1_prior: Prior for the Beta distribution for theta1.
    :param mutation_rate: Probability of mutation at each SNP location.
    :param avg_reads_num: Average number of reads per SNP location.
    :return: Dictionary containing:
        - BCR: Tensor of shape (n_cells, BCR_len, 4), one-hot encoded BCR sequences.
        - reads_counts: Tensor of shape (n_mutation, n_cells), sampled reads matrix.
        - mutation_counts: Tensor of shape (n_mutation, n_cells), mutation matrix.
        - clone_profiles: Tensor of shape (n_mutation, n_clones), mutation profiles for clones.
        - hc_clone: Tensor mapping hyperclusters to clone indices.
        - metadata: Dictionary with hypercluster and clone assignments for each cell.
        - relax_rate: Relaxation rate sampled from Beta distribution.
        - theta0: Mutation rate when no mutation occurs.
        - theta1: Mutation rate when a mutation occurs.
    """
    hiperclusters = create_hiperclusters(alpha=alpha, num_points=n_cells)

    n_hc = len(hiperclusters)
    psi = [1 / n_clones] * n_clones
    hc_clone = tf.squeeze(tf.random.categorical([psi], n_hc), axis=0)

    # Metadata
    metadata = {"hipercluster": [], "clone": []}
    for hc, (size, clone) in enumerate(zip(hiperclusters, hc_clone)):
        metadata["hipercluster"].extend([hc] * size)
        metadata["clone"].extend([clone] * size)

    # BCR
    BCR = [
        sample_hipercluster_bcr(gene_len=BCR_len, n_cells=hc, BCR_prior=BCR_prior)
        for hc in hiperclusters
    ]
    BCR = tf.concat(BCR, axis=0)

    # Reads
    reads_counts = sample_reads_matrix(
        n_snp=n_mutation, n_cells=n_cells, avg_reads_num=avg_reads_num
    )

    # Clone profiles
    clone_relax_rate = tfp.distributions.Beta(*clone_relax_prior).sample()
    clones_profiles = sample_clone_profile(
        n_snp=n_mutation,
        n_clones=n_clones,
        mutation_rate=mutation_rate,
        clone_relax_rate=clone_relax_rate,
    )

    # Mutation sampling
    theta0 = tfp.distributions.Beta(*theta0_prior).sample()
    theta1 = tfp.distributions.Beta(*theta1_prior).sample([n_mutation, 1])
    mutation_counts = sample_mutation_matrix(
        reads_counts=reads_counts,
        hiperclusters=hiperclusters,
        hc_clone=hc_clone,
        clones_profiles=clones_profiles,
        theta0=theta0,
        theta1=theta1,
    )

    # Shuffle
    permutation = tf.random.shuffle(tf.range(n_cells))

    BCR = tf.gather(BCR, permutation, axis=0)
    reads_counts = tf.gather(reads_counts, permutation, axis=1)
    mutation_counts = tf.gather(mutation_counts, permutation, axis=1)

    metadata["hipercluster"] = [metadata["hipercluster"][i] for i in permutation]
    metadata["clone"] = [metadata["clone"][i] for i in permutation]

    return {
        "BCR": BCR,
        "reads_counts": reads_counts,
        "mutation_counts": mutation_counts,
        "clone_profiles": clones_profiles,
        "hc_clone": hc_clone,
        "metadata": metadata,
        "relax_rate": clone_relax_rate,
        "theta0": theta0,
        "theta1": theta1,
    }

data = sample_data(n_cells=200)