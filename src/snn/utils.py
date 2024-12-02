from torch import cuda
from itertools import combinations
import numpy as np
import torch

def detect_device() -> cuda.device:
    device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
    return device


def all_pairs(embeddings: list) -> list:
    embedding_pairs = []
    pairs = list(combinations([x for x in embeddings], 2))
    for x, y in pairs:
        pair = (x[0], y[0], 1 if x[1] == y[1] else 0)
        embedding_pairs.append(pair)
        
    return embedding_pairs


def contrative_distances(embedding_pairs: list) -> tuple[list, list]:
    dist_0 = []
    dist_1 = []
    for embedding_1, embedding_2, target in embedding_pairs:
        if target == 0:
            dist_0.append(np.linalg.norm(embedding_1 - embedding_2))
            continue

        dist_1.append(np.linalg.norm(embedding_1 - embedding_2))
    
    return dist_0, dist_1