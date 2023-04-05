import math
from functools import partial

import re
import numpy as np
from PIL import Image

from .utils_aug import center_crop, resize


#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x):
    x /= 255
    x -= np.array((0.48145466, 0.4578275, 0.40821073))
    x /= np.array((0.26862954, 0.26130258, 0.27577711))
    return x

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def get_configs(phi):
    if phi == "openai/VIT-B-32":
        config = dict( 
            bert_type          = "openai",
            embed_dim          = 512,
            # vision
            input_resolution    = 224,
            vision_layers      = 12,
            vision_width       = 768,
            vision_patch_size  = 32,
            # text
            transformer_layers  = 12,
            transformer_width   = 512,
            transformer_heads   = 8,
            vocab_size          = 49408,
            huggingface_model_name = None
        )
    elif phi == "openai/VIT-B-16":
        config = dict( 
            bert_type          = "openai",
            embed_dim          = 512,
            # vision
            input_resolution    = 224,
            vision_layers      = 12,
            vision_width       = 768,
            vision_patch_size  = 16,
            # text
            transformer_layers  = 12,
            transformer_width   = 512,
            transformer_heads   = 8,
            vocab_size          = 49408,
            huggingface_model_name = None
        )
    elif phi == "self-cn/VIT-B-32":
        config = dict( 
            bert_type          = "huggingface",
            embed_dim          = 512,
            # vision
            input_resolution    = 224,
            vision_layers      = 12,
            vision_width       = 768,
            vision_patch_size  = 32,
            # text
            transformer_layers  = None,
            transformer_width   = None,
            transformer_heads   = None,
            vocab_size          = None,
            huggingface_model_name = "bert-base-chinese"
        )
    else:
        raise ValueError(phi + " is not support yet.")
    return config

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
# def download_weights(model_dir="./model_data"):
#     import os
#     import sys
#     import zipfile

#     from torch.hub import load_state_dict_from_url
#     try:
#         from urllib import urlretrieve
#     except ImportError:
#         from urllib.request import urlretrieve
        
#     def download_zip(url, model_dir='./pretrained'):
#         if not os.path.exists(model_dir):
#             os.makedirs(model_dir)
#         filename    = url.split('/')[-1].split('.')[0]
#         cached_file = os.path.join(model_dir, filename)
#         if not os.path.exists(cached_file):
#             os.makedirs(cached_file)
            
#             zip_file = os.path.join(model_dir, filename+'.zip')
#             sys.stderr.write('Downloading: "{}" to {}\n'.format(url, zip_file))
#             urlretrieve(url, zip_file)
#             zip_ref = zipfile.ZipFile(zip_file, 'r')
#             zip_ref.extractall(cached_file)
#             zip_ref.close()
#             os.remove(zip_file)
            
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
        
#     download_zip('https://github.com/bubbliiiing/clip-pytorch/releases/download/v1.0/chinese_wwm_ext_pytorch.zip', model_dir)
#     load_state_dict_from_url("https://github.com/bubbliiiing/clip-pytorch/releases/download/v1.0/VIT-32.pth", model_dir)