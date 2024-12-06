import torch
from snn.dataset.wrappers import ContrastiveDataset
from torchvision.datasets import VisionDataset
import numpy as np


def predict_distances(
    model: torch.nn.Module,
    dataset: ContrastiveDataset,
    device: torch.cuda.device = None,
) -> list:
    """
    Return tuples of (distance, target) for each pair in validation dataset
    """
    # detect device
    if device is None:
        cuda = torch.cuda.is_available()
        device = torch.device("cuda") if cuda else torch.device("cpu")

    distances = []
    with torch.no_grad():
        model.eval()
        for (img1, img2), target in dataset:
            input1 = img1.to(device).unsqueeze(0)
            input2 = img2.to(device).unsqueeze(0)

            embedding_1 = model.get_embedding(input1)
            embedding_2 = model.get_embedding(input2)

            dist = np.linalg.norm(
                embedding_1.squeeze().cpu().numpy()
                - embedding_2.squeeze().cpu().numpy()
            )

            distances.append((dist, target))

    return distances


def predict_embeddings(
    model: torch.nn.Module,
    dataset: VisionDataset,
    device: torch.cuda.device = None,
    return_labels: bool = False,
) -> list:
    """
    Get embedding for each image in given dataset
    """
    # detect device
    if device is None:
        cuda = torch.cuda.is_available()
        device = torch.device("cuda") if cuda else torch.device("cpu")

    embeddings = []

    with torch.no_grad():
        model.eval()
        for img, label in dataset:
            img_input = img.transpose(2, 0).transpose(2, 1).to(device).unsqueeze(0)

            embedding = model.get_embedding(img_input)
            if not return_labels:
                embeddings.append(embedding.cpu().numpy())
                continue

            embeddings.append([embedding.cpu().numpy(), label])

    return embeddings
