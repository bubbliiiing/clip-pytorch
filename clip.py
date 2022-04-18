import numpy as np
import torch
import torch.nn as nn

from nets.clip import CLIP
from utils.utils import cvtColor, preprocess_input, letterbox_image

'''
训练自己的数据集必看注释！
'''
class Clip(object):
    _defaults = {
        #-------------------------------#
        #   指向logs文件夹下的权值文件
        #-------------------------------#
        "model_path"        : 'logs\ep002-loss1.686-val_loss1.393.pth',
        #-------------------------------#
        #   特征长度
        #-------------------------------#
        "embed_dim"         : 512,
        #-------------------------------#
        #   输入图片的大小。
        #-------------------------------#
        "image_resolution"  : 224,
        #-------------------------------#
        #   文字的最大长度
        #-------------------------------#
        "context_length"    : 100,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
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
        self.net    = CLIP(embed_dim = self.embed_dim, image_resolution = self.image_resolution, context_length=self.context_length)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
            
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, captions):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = letterbox_image(image, (self.image_resolution, self.image_resolution))
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
            logits_per_image, logits_per_text = self.net(images, captions)
            
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    
        print("Label probs:", probs)
