import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

import numpy as np
from scipy.spatial import KDTree
import torch
from PIL import Image
import glob
import cv2

from SectionSeeker.snn.utils import detect_device
from SectionSeeker.snn.model import SiameseNetwork


class ImageSet:
    """
    Subscriptapble dataset-like class for loading, storing and processing image collections
    
    :param root: Path to project root directory, which contains data/image_corpus/ or data/query catalog
    :param base: Build ImageSet on top of image_corpus if True, else on top of query catalog
    :param build: Build ImageSet from filesystem instead of using saved version
    :param transform: Callable that will be applied to all images when calling __getitem__() method
    :param compatibility_mode: Convert images to PIL.Image before applying transform and returning from __getitime__() method
    :param greyscale: Load images in grayscale if True, else use 3-channel RGB
    :param normalize: If True, images will be normalized image-wise when loaded from disk
    """
    def __init__(self, 
                 root: str, 
                 base: bool = True,
                 build: bool = False, 
                 transform: Callable = None, 
                 compatibility_mode: bool = False,
                 greyscale: bool = False,
                 normalize: bool = True) -> None:
        
        self.root = root
        self.compatibility_mode = compatibility_mode
        self.greyscale = greyscale
        self.colormode = 'L' if greyscale else 'RGB'
        self.transform = transform
        self.base = base
        self.normalize = normalize
        
        if build:
            self.embeddings = []
            self.data, self.names = self._build()
            return
        
        self.data = self._load()
        
        
    def _build(self) -> Tuple[torch.Tensor, str]:

        dirpath = f"{self.root}/data/{'image_corpus' if self.base else 'query'}"
        data = []
        images = []
        names = []
        for filename in glob.glob(f"{dirpath}/*png"):
            im = Image.open(filename)
            # resize into common shape
            im = im.convert(self.colormode).resize((118, 143))
            if self.normalize:
                im = cv2.normalize(np.array(im), None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32FC1)
            image = np.array(im, dtype=np.float32)    
            fname = filename.split('/')[-1]
            data.append(image)
            names.append(fname)
        return torch.from_numpy(np.array(data)), names
        
    def _load(self) -> Tuple[torch.Tensor, str]:
        ...
        
    def save(self) -> None:
        ...
        
    def build_embeddings(self, model: SiameseNetwork, device: torch.cuda.device = None):
        
        if device is None:
            device = detect_device()
        
        with torch.no_grad():
            model.eval()
            for img, name in self:
                img_input = img.transpose(2,0).transpose(2,1).to(device).unsqueeze(0)
                embedding = model.get_embedding(img_input)
                self.embeddings.append((embedding, name))
                
        return self
        
    def get_embeddings(self) -> List[Tuple[torch.Tensor, str]]:
        if self.embeddings is None:
            raise RuntimeError('Embedding collection is empty. Run self.build_embeddings() method to build it')
        
        return self.embeddings
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        name = self.names[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.compatibility_mode:
            img = Image.fromarray(img.numpy(), mode=self.colormode)

        if self.transform is not None:
            img = self.transform(img)

        return img, name
      
    
class SearchTree:
    """
    Wrapper for k-d tree built on image embeddings
    
    :param query_set: instance of base ImageSet with built embedding representation
    """
    def __init__(self, query_set: ImageSet) -> None:
        embeddings = query_set.get_embeddings()
        self.embeddings = np.concatenate([x[0].cpu().numpy() for x in embeddings], axis=0)
        self.names = np.array([x[1] for x in embeddings])
        self.kdtree = self._build_kdtree()
        
    def _build_kdtree(self) -> KDTree:
        print('Building KD-Tree from embeddings')
        return KDTree(self.embeddings)
        
    def query(self, anchors: ImageSet, k: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of provided anchor embeddings
        
        :param anchors: instance of query (reference) ImageSet with built embedding representation
        
        :returns: tuple of reference_labels, distances to matched label embeddings, matched label embeddings, matched_labels 
        """
        
        reference = anchors.get_embeddings()
        reference_embeddings = np.concatenate([x[0].cpu().numpy() for x in reference], axis=0)
        reference_labels = np.array([x[1] for x in reference])
        
        distances, indices = self.kdtree.query(reference_embeddings, k=k, workers=-1)          
        return reference_labels, distances, self.embeddings[indices], self.names[indices]
    
    def __call__(self, *args, **kwargs) -> Any:
        return self.query(*args, **kwargs)
        
