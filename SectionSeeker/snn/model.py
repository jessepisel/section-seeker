from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.models import get_model, get_weight
from torch.optim.lr_scheduler import StepLR
from typing import Mapping
from .config import ModelConfig, defaultConfig

class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation can take different vision models to generate features from images - model and specific checkpoint can be specified with the config file. 
        
        :param config: model configuration mapping
    """
    def __init__(self, config: Mapping = defaultConfig):
        super().__init__()

        self.config = config
        weights_loaded = False
        if self.config.BACKBONE_MODEL_WEIGHTS:
            # configure model
            try:
                model_weights = get_weight(self.config.BACKBONE_MODEL_WEIGHTS)
                self.backbone = get_model(self.config.BACKBONE_MODEL, weights=model_weights)
                weights_loaded = True
            except ValueError:
                self.backbone = get_model(self.config.BACKBONE_MODEL)
                self.backbone.load_state_dict(torch.load(self.config.BACKBONE_MODEL_WEIGHTS))
                weights_loaded = True
        
        if self.config.FC_IN_FEATURES < 1:
            self.fc_in_features = self.backbone.fc.in_features
        else:
            self.fc_in_features = self.config.FC_IN_FEATURES
            
        self.latent_space_dim = self.config.LATENT_SPACE_DIM
        
        # check if the backbone is ViT
        self.vit = 'vit' in self.config.BACKBONE_MODEL.lower()
        
        if self.vit:
            self.feature_extractor = nn.Sequential(*(list(self.backbone.children())[:-1]))
            self.encoder = self.feature_extractor[1]
            
        else:
             # remove the last layer of the model
            self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))
        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.latent_space_dim),
        )
        
        # initialize the weights
        if not weights_loaded:
            print('Initialized weights for backbone model')
            self.backbone.apply(self.init_weights)
        print('Initialized weights for fully connected component')
        self.fc.apply(self.init_weights)
    
    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        if self.vit:
            self.feature_extractor.requires_grad_(False)
            self.encoder.requires_grad_(False)
    
    def unfreeze_backbone(self):
        self.backbone.requires_grad_(True)
        if self.vit:
            self.feature_extractor.requires_grad_(True)
            self.encoder.requires_grad_(True)
        
    def _vit_forward(self, x):
        x_processed = self.backbone._process_input(x)

        n = x_processed.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x_processed = torch.cat([batch_class_token, x_processed], dim=1)
        x_features = self.encoder(x_processed)
        return x_features[:, 0]
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, input1, input2):
        # pass forward both input images
        # get two images' features
        output1 = self.get_embedding(input1)
        output2 = self.get_embedding(input2)

        return output1, output2
    
    def get_embedding(self, x):
        """ Return embedding of single inmpt"""
        if self.vit:
            output = self._vit_forward(x)
        else:
            output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)
    
    
class ClassificationNet(nn.Module):
    """
    Binary classifier built on top of embedding network. 
    
    :param embedding_net: neural network (toch.nn.Module) that will provide embeddings for input images. Must have latent_space_dim attribute and 
                            return tuple of embeddings for two input images on forward call
    """
    def __init__(self, embedding_net: nn.Module):
            
        super().__init__()
        self.embedding_net = embedding_net
        
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_net.latent_space_dim * 2, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, input1, input2):
        embedding_1, embedding_2 = self.embedding_net(input1, input2)
        output = self.classifier(torch.cat([embedding_1, embedding_2], dim=0))
        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        return output