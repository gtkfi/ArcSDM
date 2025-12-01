
import random
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
    synthetic_samples = []
    sample_ids = list(range(n_minority))
    for i in range(n_synthetic):
        if len(sample_ids) == 0:
                sample_ids = list(range(n_minority))
        random_id = sample_ids.pop(random.randrange(len(sample_ids)))
        chosen_sample = minority_X[random_id]
        chosen_neighbor = minority_X[rng.choice(neighbor_list[random_id])]
       
        difference = chosen_neighbor - chosen_sample
        gap = rng.random()
 
        synthetic_samples.append(chosen_sample + gap * difference)
 
    final_X = np.concatenate([X, np.array(synthetic_samples)])
    final_y = np.concatenate([y, np.repeat(minority_class, len(synthetic_samples))])
 
    return final_X, final_y