from typing import Any, Union
from pytorch_lightning import LightningModule
from torch import (
    nn,
    optim,
)
from transformers import (
    AutoTokenizer
)
from vision_coder.models.vision_encoder import (
    VisionCoderImageProcessor,
    VisionCoderVisionEncoder
)
from vision_coder.models.text_decoder import (
    LLamaVisionCoder
)

from vision_coder.models.vision_coder import (
    VisionCoder
)


class VisionCoderTrainer(LightningModule):
    def __init__(
        self,
        text_decoder: Union[str, LLamaVisionCoder],
        vision_encoder: Union[str, VisionCoderVisionEncoder],
        tokenizer: Union[str, AutoTokenizer],
        image_processor = Union[str, VisionCoderImageProcessor],
        learning_rate: float = 1e-4,
        warmup_steps: int = 2000,
        beta1: float = 0.99,
        beta2: float = 0.98,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(text_decoder, str):
            text_decoder = LLamaVisionCoder.from_pretrained(text_decoder)

        if isinstance(vision_encoder, str):
            vision_encoder = VisionCoderVisionEncoder.from_pretrained(vision_encoder)

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if isinstance(image_processor, str):
            self.image_processor = VisionCoderImageProcessor.from_pretrained(image_processor)

        self.vision_coder = VisionCoder(
            vision_encoder=vision_encoder,
            text_decoder=text_decoder,
            image_token_id=self.tokenizer.image_token_id
        )
        
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.vision_coder.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=None)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
    
    def training_step(self, *args: Any, **kwargs: Any):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)