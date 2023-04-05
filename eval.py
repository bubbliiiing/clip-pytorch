import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip import CLIP
from utils.dataloader import ClipDataset, dataset_collate
from utils.metrics import itm_eval


if __name__ == "__main__":
    #------------------------------------------------------#
    #   datasets_path           数据集路径
    #   datasets_val_json_path  验证样本的标签
    #   batch_size              验证的batch_size
    #------------------------------------------------------#
    datasets_path               = "datasets/"
    datasets_val_json_path      = "datasets/cn_val.json"
    batch_size                  = 64
    num_workers                 = 4

    # 创建模型
    model       = CLIP()

    # 计算样本数
    val_lines   = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
    num_val     = len(val_lines)
    # 创建数据集
    val_dataset = ClipDataset([model.config['input_resolution'], model.config['input_resolution']], val_lines, datasets_path, random = False)
    gen_val     = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=False, collate_fn=dataset_collate, sampler=None)

    # 获得视觉特征和文本特征
    i_features = []
    t_features = []
    for iteration, batch in tqdm(enumerate(gen_val)):
        images, texts = batch
        with torch.no_grad():
            if model.cuda:
                images  = images.cuda()
            
            images_feature, _ = model.detect_image_for_eval(images, texts=None)
            i_features.append(images_feature)

    texts       = gen_val.dataset.text
    num_text    = len(texts)
    for i in tqdm(range(0, num_text, batch_size)):
        text = texts[i: min(num_text, i + batch_size)]
        with torch.no_grad():
            _, texts_feature = model.detect_image_for_eval(images=None, texts=text)
            t_features.append(texts_feature)

    i_features = torch.cat(i_features, 0)
    t_features = torch.cat(t_features, 0)
    
    i_features  = i_features / i_features.norm(dim=-1, keepdim=True)
    t_features  = t_features / t_features.norm(dim=-1, keepdim=True)

    logits_per_image    = i_features @ t_features.t()
    logits_per_text     = logits_per_image.t()

    logits_per_image    = logits_per_image.cpu().numpy()
    logits_per_text     = logits_per_text.cpu().numpy()
    
    print(itm_eval(logits_per_image, logits_per_text, gen_val.dataset.txt2img, gen_val.dataset.img2txt))