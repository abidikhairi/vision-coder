import torch
import unittest
from transformers import (
    ViTConfig,
    LlamaConfig,
)
from vision_coder.models.vision_coder import VisionCoder
from vision_coder.models.vision_encoder import VisionCoderVisionEncoder
from vision_coder.models.text_decoder import LLamaVisionCoder


class TestVisionCoder(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_init_from_pretrained_models(self):
        vision_encoder = VisionCoderVisionEncoder(ViTConfig(hidden_size=128, num_hidden_layers=2, intermediate_size=256, num_attention_heads=4))
        text_decoder = LLamaVisionCoder(LlamaConfig(vocab_size=2048, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4))
        model = VisionCoder.from_pretrained_models(
            vision_encoder,
            text_decoder,
            -1
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.vision_encoder)
        self.assertIsNotNone(model.text_decoder)
        self.assertEqual(-1, model.config.image_token_id)

    # def test_save_load_pretrained(self):
    #     tokenizer = AutoTokenizer.from_pretrained('data/tokenizer')
    #     model = VisionCoder.from_pretrained_models(
    #         './data/vision_encoder',
    #         './data/text_decoder',
    #         tokenizer.image_token_id
    #     )
        
    #     # TODO: fix save_pretrained
    #     model.save_pretrained('data/vision-coder')
    #     model = VisionCoder.from_pretrained('data/vision-coder')
        
    #     self.assertIsNotNone(model)
    #     self.assertIsNotNone(model.vision_encoder)
    #     self.assertIsNotNone(model.text_decoder)
    #     self.assertEqual(tokenizer.image_token_id, model.config.image_token_id)

    def test_forward(self):
        # TODO: need to configure model before
        self.assertTrue(True)
    