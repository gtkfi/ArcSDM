
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(
    X,
    y,
    n_synthetic=None,
    minority_class=1,
    k_neighbors=5,
    random_state=None
):
    """
    Perform synthetic minority oversampling (SMOTE). Implements the SMOTE algorithm described in:
    Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
    SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357.
    Differs from the original algorithm by offering ability to do oversampling by defining the number of wanted synthetic
    samples instead using just integral multiples of the minority class size for oversampling.
 
    Args:
        X: Input data. 2D numpy array where rows represent samples and columns represent features.
        y: Target labels. Should be 1D binary array of zeros and ones.
        n_synthetic: Number of synthetic samples to be generated. If None, balanced 50-50 dataset is created.
        minority_class: Class label of the minority class. Defaults to 1.
        k_neighbors: Number of neighbours used for SMOTE. Defaults to 5.
        random_state: Integer number can be used as seed for reproductibility.
    Returns:
        final_X: Numpy array of input data with synthetic samples added.
        final_y: Numpy array of target labels with synthetic samples added.
    """
    n_minority = (y==minority_class).sum()
    if n_minority < 2:
        raise ValueError("There are less than 2 minority class samples")
   
    if k_neighbors > n_minority:
        k_neighbors = n_minority
 
    n_majority = len(y) - n_minority
 
    if n_minority == n_majority and n_synthetic is None:
        raise ValueError("Dataset is already balanced and n_synthetic is not defined")
 
    if n_synthetic is None:
        n_synthetic = int(n_majority - n_minority)
 
    rng = np.random.default_rng(random_state)
 
    minority_X = X[np.where(y == minority_class)]
   
    # Extra neighbour is needed as the NearestNeighbors considers a sample itself as a neighbour
    # The extra neighbour is removed later by slicing the neighbour list
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(minority_X)
    neighbor_list = nn.kneighbors(minority_X, return_distance=False)[:, 1:]
 
    # Generate synthetic samples by using sampling without replacement to select seed points
    # Sampling begins again with full list after all minority class samples are used.
    # Vectorized synthetic sample generation
    # 1. Select random minority samples (with replacement if needed)
    sample_ids = np.arange(n_minority)
    n_repeats = int(np.ceil(n_synthetic / n_minority))
    all_seed_ids = np.tile(sample_ids, n_repeats)[:n_synthetic]
    # 2. For each seed, select a random neighbor
    neighbor_choices = [rng.choice(neighbor_list[idx]) for idx in all_seed_ids]
    chosen_samples = minority_X[all_seed_ids]
    chosen_neighbors = minority_X[neighbor_choices]
    # 3. Generate all gap values
    gaps = rng.random(n_synthetic)
    # 4. Compute synthetic samples
    synthetic_samples = chosen_samples + gaps[:, None] * (chosen_neighbors - chosen_samples)
 
    # Only add synthetic samples to the specified minority class, keep all original data
    final_X = np.concatenate([X, synthetic_samples])
    final_y = np.concatenate([y, np.full(len(synthetic_samples), minority_class, dtype=y.dtype)])
    return final_X, final_y