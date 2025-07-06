from transformers import (
    ViTModel, ViTImageProcessor
)
from transformers.image_utils import ChannelDimension

class VisionCoderVisionEncoder(ViTModel):
    """
    A wrapper around HuggingFace's ViTModel to allow for customized forward passes.

    Inherits:
        ViTModel: Vision Transformer model from HuggingFace Transformers.

    Args:
        config (PretrainedConfig): The configuration object for the ViT model.
        add_pooling_layer (bool, optional): Whether to add a pooling layer at the end. Defaults to True.
        use_mask_token (bool, optional): Whether to use mask tokens. Defaults to False.

    Forward Args:
        pixel_values (torch.Tensor): Input image tensor of shape (B, C, H, W).
        bool_masked_pos (torch.BoolTensor, optional): Mask for token masking.
        head_mask (torch.Tensor, optional): Mask to nullify selected heads.
        output_attentions (bool, optional): Whether to return attention weights.
        output_hidden_states (bool, optional): Whether to return hidden states.
        interpolate_pos_encoding (bool, optional): Whether to interpolate positional encodings.
        return_dict (bool, optional): Whether to return outputs as a dict.

    Returns:
        BaseModelOutputWithPooling: Output object containing last hidden states, pooled output, etc.
    """
    def __init__(self, config, add_pooling_layer = True, use_mask_token = False):
        super().__init__(config, add_pooling_layer, use_mask_token)
    
    def forward(self, pixel_values = None, bool_masked_pos = None, head_mask = None, output_attentions = None, output_hidden_states = None, interpolate_pos_encoding = None, return_dict = None):
        return super().forward(pixel_values, bool_masked_pos, head_mask, output_attentions, output_hidden_states, interpolate_pos_encoding, return_dict)
    
class VisionCoderImageProcessor(ViTImageProcessor):
    """
    Custom image processor for preprocessing input images for VisionCoderVisionEncoder.

    Inherits:
        ViTImageProcessor: Image processor from HuggingFace for ViT models.

    Args:
        images (PIL.Image or np.ndarray or torch.Tensor): Input images.
        do_resize (bool, optional): Whether to resize the image.
        size (dict or tuple, optional): Target size.
        resample (int, optional): Resampling filter.
        do_rescale (bool, optional): Whether to rescale the image.
        rescale_factor (float, optional): Factor to rescale pixel values.
        do_normalize (bool, optional): Whether to normalize the image.
        image_mean (list, optional): Mean for normalization.
        image_std (list, optional): Standard deviation for normalization.
        return_tensors (str, optional): Format to return tensors ('pt', 'np', etc.).
        data_format (ChannelDimension, optional): Channel format of output tensor.
        input_data_format (str, optional): Format of input data.
        do_convert_rgb (bool, optional): Whether to convert image to RGB.

    Returns:
        dict: Dictionary containing preprocessed image tensors.
    """
    def preprocess(self, images, do_resize = None, size = None, resample = None, do_rescale = None, rescale_factor = None, do_normalize = None, image_mean = None, image_std = None, return_tensors = None, data_format = ChannelDimension.FIRST, input_data_format = None, do_convert_rgb = None):
        return super().preprocess(images, do_resize, size, resample, do_rescale, rescale_factor, do_normalize, image_mean, image_std, return_tensors, data_format, input_data_format, do_convert_rgb)
