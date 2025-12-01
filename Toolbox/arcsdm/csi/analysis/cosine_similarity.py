import numpy as np

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True, cache=True)
def cosine_similarity_numba(vector_a, vector_b, nodata_value):
    dot_sum = 0.0
    norm_a = 0.0
    norm_b = 0.0
    count = 0
    for i in range(len(vector_a)):
        a_val = vector_a[i]
        b_val = vector_b[i]
        if (np.isnan(a_val) or np.isnan(b_val) or a_val == nodata_value or b_val == nodata_value):
            continue
        dot_sum += a_val * b_val
        norm_a += a_val * a_val
        norm_b += b_val * b_val
        count += 1
    if count == 0 or norm_a == 0.0 or norm_b == 0.0:
        return nodata_value
    return dot_sum / (np.sqrt(norm_a) * np.sqrt(norm_b))

def cosine_similarity(vector_a, vector_b, nodata_value=-9999.0):
    """
    Compute the cosine similarity between two vectors, ignoring nodata values.
    Returns:
        nodata_value if no valid comparison can be made.
    """
    if HAS_NUMBA:
        return cosine_similarity_numba(vector_a, vector_b, nodata_value)
    else:
        mask = ~(
            np.isnan(vector_a) | np.isnan(vector_b) |
            (vector_a == nodata_value) | (vector_b == nodata_value)
        )
        if not np.any(mask):
            return nodata_value
        a_clean = vector_a[mask]
        b_clean = vector_b[mask]
        if a_clean.size == 0:
            return nodata_value
        dot_product = np.dot(a_clean, b_clean)
        magnitude_a = np.linalg.norm(a_clean)
        magnitude_b = np.linalg.norm(b_clean)
        if magnitude_a == 0 or magnitude_b == 0:
            return nodata_value
        return dot_product / (magnitude_a * magnitude_b)
