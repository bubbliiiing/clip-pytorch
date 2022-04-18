import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from .vit import LayerNorm, VisionTransformer


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim          = 512,
                 # vision
                 image_resolution   = 224,
                 vision_layers      = 12,
                 vision_width       = 768,
                 vision_patch_size  = 32,
                 # text
                 context_length     = 100,
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution    = image_resolution,
            patch_size          = vision_patch_size,
            width               = vision_width,
            layers              = vision_layers,
            heads               = vision_heads,
            output_dim          = embed_dim
        )
        self.visual.load_state_dict(torch.load("model_data/VIT-B-32-Only.pth"))

        self.tokenizer          = BertTokenizer.from_pretrained("model_data/chinese_wwm_ext_pytorch")
        self.text               = BertModel.from_pretrained("model_data/chinese_wwm_ext_pytorch")
        transformer_width       = self.text.config.hidden_size
        self.ln_final           = LayerNorm(transformer_width)
        self.text_projection    = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
            
        self.logit_scale        = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.tokenizer(text, return_tensors="pt", padding=True)
        if self.visual.conv1.weight.is_cuda:
            input_ids       = x.input_ids.cuda()
            attention_mask  = x.attention_mask.cuda()
            token_type_ids  = x.token_type_ids.cuda()
        else:
            input_ids       = x.input_ids
            attention_mask  = x.attention_mask
            token_type_ids  = x.token_type_ids
        x = self.text(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection

        return x

    def forward(self, image, text):
        image_features  = self.encode_image(image)
        text_features   = self.encode_text(text)

        image_features  = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features   = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale         = self.logit_scale.exp()
        logits_per_image    = logit_scale * image_features @ text_features.t()
        logits_per_text     = logits_per_image.t()

        return logits_per_image, logits_per_text

