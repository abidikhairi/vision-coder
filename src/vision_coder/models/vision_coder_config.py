from transformers.models import (
    LlamaConfig,
    ViTConfig
)
from transformers.configuration_utils import PretrainedConfig


class VisionCoderConfig(PretrainedConfig):
    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        image_token_id: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.text_config = text_config if text_config else LlamaConfig()
        self.vision_config = vision_config if vision_config else ViTConfig()
        
        self.image_token_id = image_token_id
        
        self.vision_config_type = vision_config.__class__.__name__
        self.text_config_type = text_config.__class__.__name__
    
    @classmethod
    def from_vision_text_configs(
        cls,
        text_config: LlamaConfig,
        vision_config: ViTConfig,
        **kwargs
    ) -> "VisionCoderConfig":
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs
        )
    
    def to_dict(self):
        output = super().to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        return output
    
    @classmethod
    def from_dict(cls, config_dict):
        text_config = LlamaConfig.from_dict(config_dict.pop("text_config"))
        vision_config = ViTConfig.from_dict(config_dict.pop("vision_config"))
        
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            **config_dict
        )
