from torch import nn
import torch
from transformers import (
    LlamaForCausalLM
)


class LLamaVisionCoder(LlamaForCausalLM):
    """
    A wrapper around HuggingFace's LlamaForCausalLM for possible future extension or customization.

    Inherits:
        LlamaForCausalLM: Causal language model based on the LLaMA architecture.

    Args:
        config (PretrainedConfig): The configuration object for the LLaMA model.

    Forward Args:
        input_ids (torch.LongTensor, optional): Input token IDs.
        attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens.
        position_ids (torch.LongTensor, optional): Position IDs for input tokens.
        past_key_values (tuple, optional): Cached past key/values for faster decoding.
        inputs_embeds (torch.FloatTensor, optional): Precomputed input embeddings.
        labels (torch.LongTensor, optional): Target labels for language modeling.
        use_cache (bool, optional): Whether to use caching.
        output_attentions (bool, optional): Whether to return attention weights.
        output_hidden_states (bool, optional): Whether to return all hidden states.
        cache_position (torch.LongTensor, optional): Custom cache indexing position.
        logits_to_keep (int, optional): If > 0, keeps only top-k logits for decoding.
        **kwargs: Additional arguments passed to the base model.

    Returns:
        CausalLMOutputWithPast: Outputs including logits, past_key_values, etc.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids = None, attention_mask = None, position_ids = None, past_key_values = None, inputs_embeds = None, labels = None, use_cache = None, output_attentions = None, output_hidden_states = None, cache_position = None, logits_to_keep = 0, **kwargs):
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **kwargs)


class VisionToTextProjection(nn.Module):
    """
    Projects image embeddings from vision encoder space to text decoder embedding space.

    Args:
        vision_embed_size (int): Dimensionality of the vision encoder's output embeddings.
        text_embed_size (int): Dimensionality of the text decoder's input embeddings.

    Attributes:
        linear (nn.Linear): Linear layer without bias for projecting embeddings.

    Forward Args:
        image_embeds (torch.Tensor): Tensor of shape (B, D_v), where D_v = vision_embed_size.

    Returns:
        torch.Tensor: Projected embeddings of shape (B, D_t), where D_t = text_embed_size.
    """
    def __init__(self, vision_embed_size: int, text_embed_size: int):
        super().__init__()
        self.linear = nn.Linear(vision_embed_size, text_embed_size, False)
        
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.linear(image_embeds)
