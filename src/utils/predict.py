import sys
sys.path.append('..')
import torch
from src.snn.utils import detect_device
import numpy as np

def predict_embeddings(model: torch.nn.Module, 
                       data: torch.Tensor, 
                       device: torch.cuda.device = None) -> list:
    """
    Get embedding for each image in given dataset
    """
    #detect device
    if device is None:
        device = detect_device()
        
    embeddings = []
    
    with torch.no_grad():
        model.eval()
        for img in data:
            img_input = img.transpose(2,0).transpose(2,1).to(device).unsqueeze(0)

            embedding = model.get_embedding(img_input)
                
            embeddings.append([embedding.cpu().numpy(), label])
    
    return embeddings