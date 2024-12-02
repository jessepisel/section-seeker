from dataclasses import dataclass

@dataclass
class ModelConfig:
    BACKBONE_MODEL: str = 'ResNet50'
    BACKBONE_MODEL_WEIGHTS: str = 'ResNet50_Weights.IMAGENET1K_V2'
    LATENT_SPACE_DIM: int = 8
    FC_IN_FEATURES: int = -1
    
    
defaultConfig = ModelConfig()

vitBaseConfig = ModelConfig(BACKBONE_MODEL = 'ViT_B_16',
    BACKBONE_MODEL_WEIGHTS = 'ViT_B_16_Weights.DEFAULT',
    LATENT_SPACE_DIM = 16,
    FC_IN_FEATURES = 768)

vitBaseConfigPretrained = ModelConfig(BACKBONE_MODEL = 'ViT_B_16',
    BACKBONE_MODEL_WEIGHTS = '../checkpoints/ViT_B_16_SEISMIC_SGD_28G_M75.pth',
    LATENT_SPACE_DIM = 16,
    FC_IN_FEATURES = 768)