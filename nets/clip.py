import numpy as np
import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from transformers import BertModel, BertTokenizer

from .vit import VisionTransformer


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim          = 512,
                 # vision
                 input_shape        = [224, 224],
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
            input_shape         = input_shape,
            patch_size          = vision_patch_size,
            num_features        = vision_width,
            depth               = vision_layers,
            num_heads           = vision_heads,
        )
        self.visual.load_state_dict(load_state_dict_from_url("https://github.com/bubbliiiing/clip-pytorch/releases/download/v1.0/VIT-32.pth", model_dir="model_data"), strict=False)
        self.visual.head = nn.Linear(self.visual.num_features, embed_dim)

        self.tokenizer          = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", cache_dir='model_data')
        self.text               = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext", cache_dir='model_data')
        transformer_width       = self.text.config.hidden_size
        self.ln_final           = nn.LayerNorm(transformer_width)
        self.text_projection    = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
            
        self.logit_scale        = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype
    

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids       = x.input_ids.to(self.visual.patch_embed.proj.weight.device)
        attention_mask  = x.attention_mask.to(self.visual.patch_embed.proj.weight.device)
        token_type_ids  = x.token_type_ids.to(self.visual.patch_embed.proj.weight.device)
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

