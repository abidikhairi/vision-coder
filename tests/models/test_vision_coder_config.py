import unittest
from transformers import (
    LlamaConfig, ViTConfig
)
from vision_coder.models.vision_coder_config import VisionCoderConfig


class TestVisionCoderConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.image_token_id = -1
        return super().setUp()
    
    def test_create_from_configs(self):
        text_config = LlamaConfig()
        vision_config = ViTConfig()
        
        config = VisionCoderConfig(text_config, vision_config, self.image_token_id)
        
        self.assertEqual(-1, config.image_token_id)
        self.assertIsNotNone(config.text_config)
        self.assertIsNotNone(config.vision_config)
        
        