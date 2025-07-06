from typing import Optional, Union
import torch
from transformers.modeling_utils import PreTrainedModel

from vision_coder.models.vision_encoder import VisionCoderVisionEncoder
from vision_coder.models.vision_coder_config import VisionCoderConfig
from vision_coder.models.text_decoder import (
    LLamaVisionCoder,
    VisionToTextProjection
)


class VisionCoder(PreTrainedModel):
    def __init__(
        self,
        config: Optional[VisionCoderConfig] = None,
        vision_encoder: PreTrainedModel = None, # type: ignore
        text_decoder: PreTrainedModel = None, # type: ignore
        image_token_id: int = None # type: ignore
    ):
        
        if config is None:
            if vision_encoder is None or text_decoder is None or image_token_id is None:
                raise ValueError(
                    "Must provide either a config OR vision_encoder, text_decoder, and image_token_id"
                )
            config = VisionCoderConfig(
                vision_config=vision_encoder.config,
                text_config=text_decoder.config,
                image_token_id=image_token_id
            )
        else:
            if image_token_id is not None:
                config.image_token_id = image_token_id

        super().__init__(config)
        
        if vision_encoder is not None:
            self.vision_encoder = vision_encoder
        else:
            self.vision_encoder = VisionCoderVisionEncoder(config.vision_config)
        
        if text_decoder is not None:
            self.text_decoder = text_decoder
        else:
            self.text_decoder = LLamaVisionCoder(config.text_config)
        
        if config.image_token_id is None:
            raise ValueError("image_token_id must be set in config or constructor")
        
        self.vision_proj = VisionToTextProjection(
            vision_embed_size=self.vision_encoder.config.hidden_size,
            text_embed_size=self.text_decoder.config.hidden_size
        )
    
    
    @classmethod
    def from_pretrained_models(
        cls,
        vision_encoder_path_or_id: Union[str, VisionCoderVisionEncoder],
        text_decoder_path_or_id: Union[str, LLamaVisionCoder],
        image_token_id: int
    ):
        vit_model = VisionCoderVisionEncoder.from_pretrained(vision_encoder_path_or_id) if isinstance(vision_encoder_path_or_id, str) else vision_encoder_path_or_id
        llama_model = LLamaVisionCoder.from_pretrained(text_decoder_path_or_id) if isinstance(text_decoder_path_or_id, str) else text_decoder_path_or_id
      
        return cls(
            vision_encoder=vit_model,
            text_decoder=llama_model,
            image_token_id=image_token_id
        )
    
        
    def forward(
        self,
        pixel_values = None,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        cache_position = None,
        logits_to_keep = 0,
        **kwargs
    ):
        # [bs, seq_len, hidden_size]
        text_features = self.text_decoder.model.embed_tokens(input_ids) # type: ignore
        
        # [bs, vision_hidden_size]
        image_features = self.vision_encoder(pixel_values=pixel_values, return_dict=True).pooler_output
        
        # [bs, hidden_size]
        image_features = self.vision_proj(image_features)
        
        # [bs, seq_len]    
        image_token_mask = (input_ids == self.config.image_token_id)
    
        # [bs, 1, hidden_size]
        projected_image = image_features.unsqueeze(1)
    
        # Scatter image features into text features
        inputs_embeds = torch.where(
            image_token_mask.unsqueeze(-1),
            projected_image.expand_as(text_features),
            text_features
        )        
        
        # TODO: loss should not be calculated in forward
        if labels is not None:
            labels = labels.clone()
            labels[image_token_mask] = -100  # Ignore index for loss
            
        return self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
