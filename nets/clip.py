import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from .bert import Transformer
from .simple_tokenizer import SimpleTokenizer, tokenize
from .vit import VisionTransformer


class CLIP(nn.Module):
    def __init__(
        self,
        bert_type           = "openai",

        embed_dim          = 512,
        # vision
        input_resolution   = 224,
        vision_layers      = 12,
        vision_width       = 768,
        vision_patch_size  = 32,
        # text
        context_length      = 77,
        transformer_layers  = 12,
        transformer_width   = 768,
        transformer_heads   = 12,
        vocab_size          = 49408,
        **kwargs
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads    = vision_width // 64
        self.visual     = VisionTransformer(
            input_resolution    = input_resolution,
            patch_size          = vision_patch_size,
            width               = vision_width,
            layers              = vision_layers,
            heads               = vision_heads,
            output_dim          = embed_dim
        )

        self.bert_type = bert_type
        if bert_type == "openai":
            self.tokenizer          = SimpleTokenizer()
            self.transformer        = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.vocab_size             = vocab_size
            self.token_embedding        = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding   = nn.Parameter(torch.empty(self.context_length, transformer_width))
        elif bert_type == "huggingface":
            self.tokenizer          = BertTokenizer.from_pretrained(kwargs['huggingface_model_name'])
            self.transformer        = BertModel.from_pretrained(kwargs['huggingface_model_name'])
            transformer_width       = self.transformer.config.hidden_size

        self.text_projection        = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        self.ln_final               = nn.LayerNorm(transformer_width)
        self.logit_scale            = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        if self.bert_type == "openai":
            text = tokenize(self.tokenizer, text).to(self.visual.conv1.weight.device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        elif self.bert_type == "huggingface":
            x = self.tokenizer(text, return_tensors="pt", padding=True)
            input_ids       = x.input_ids.to(self.visual.conv1.weight.device)
            attention_mask  = x.attention_mask.to(self.visual.conv1.weight.device)
            token_type_ids  = x.token_type_ids.to(self.visual.conv1.weight.device)
            x = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
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
    