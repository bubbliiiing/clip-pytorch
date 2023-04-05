import numpy as np
import torch

from nets.clip import CLIP as CLIP_Model
from utils.utils import (cvtColor, get_configs, letterbox_image,
                         preprocess_input)

'''
训练自己的数据集必看注释！
'''
class CLIP(object):
    _defaults = {
        #--------------------------------------------------------------------#
        #   指向logs文件夹下的权值文件
        #--------------------------------------------------------------------#
        "model_path"        : 'model_data/ViT-B-16-OpenAI.pth',
        #----------------------------------------------------------------------------------------------------------------------------------------#
        #   模型的种类
        #   openai/VIT-B-16为openai公司开源的CLIP模型中，VIT-B-16规模的CLIP模型，英文文本与图片匹配，有公开预训练权重可用。
        #   openai/VIT-B-32为openai公司开源的CLIP模型中，VIT-B-32规模的CLIP模型，英文文本与图片匹配，有公开预训练权重可用。
        #   self-cn/VIT-B-32为自实现的模型，VIT-B-32规模的CLIP模型，英文文本与图片匹配，中文文本与图片匹配，无公开预训练权重可用
        #-----------------------------------------------------------------------------------------------------------------------------------------#
        "phi"               : "openai/VIT-B-16",
        #--------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
        #--------------------------------------------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------------------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化CLIP
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        self.config = get_configs(self.phi)

        self.net    = CLIP_Model(**self.config)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            # self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
            
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, texts):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   CenterCrop
        #---------------------------------------------------------#
        image_data  = letterbox_image(image, [self.config['input_resolution'], self.config['input_resolution']], self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            logits_per_image, logits_per_text = self.net(images, texts)
            
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        return probs
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image_for_eval(self, images=None, texts=None):
        with torch.no_grad():
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            if images is not None:
                images_feature = self.net.encode_image(images)
            else:
                images_feature = None
                
            if texts is not None:
                texts_feature = self.net.encode_text(texts)
            else:
                texts_feature = None
        
        return images_feature, texts_feature