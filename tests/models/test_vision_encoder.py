import unittest
from PIL import Image
import requests

from vision_coder.models.vision_encoder import (
    VisionCoderVisionEncoder,
    VisionCoderImageProcessor
)


class TestVisionCoderVisionEncoder(unittest.TestCase):
    def setUp(self):
        self.model = VisionCoderVisionEncoder.from_pretrained('google/vit-base-patch16-224')
        self.processor = VisionCoderImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.dummy_image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw)
    
    def test_image_processor(self):
        inputs = self.processor(images=self.dummy_image, return_tensors="pt")

        self.assertIsNotNone(inputs)
        self.assertEqual(1, len(inputs))
        self.assertTrue('pixel_values' in inputs)
        
    def test_output_shapes(self):
        inputs = self.processor(images=self.dummy_image, return_tensors="pt")
        outputs = self.model(**inputs)
        self.assertEqual((1, 768), outputs.pooler_output.shape)
