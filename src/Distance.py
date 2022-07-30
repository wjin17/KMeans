import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))