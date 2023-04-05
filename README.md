## CLIP：Learning Transferable Visual Models From Natural Language Supervision 在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [训练步骤 How2train](#训练步骤)
5. [预测步骤 How2predict](#预测步骤)
6. [评估步骤 How2eval](#评估步骤)
7. [参考资料 Reference](#Reference)

## Top News
**`2023-04`**:**创建仓库，支持训练中文数据集、英文数据集、支持评估等。**   

### 所需环境
torch==1.7.1以上

### 文件下载  
训练所需的pth可以在百度网盘下载。       
链接: https://pan.baidu.com/s/1b9Nt-UuqOJfhbhJYVyrK0g     
提取码: mfnc     

flickr8k数据集下载地址如下，里面已经包括了训练集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1UzaGmbEGz1BXZ0IXK1TT7g     
提取码: exg3    

## 训练步骤
### a、训练flickr8k数据集
1. 数据集的准备   
**本文使用flickr8k数据集，解压后放在datasets中**  
flickr8k数据集由数据图片与标注文件组成，数据图片位于flickr8k-images中，为图片文件。
标注文件为*.json文件，*.json的格式如下，image为图片的路径，caption为对应的文本，为一个列表，内容可以多条也可以单条：
```python
[
  {
    "image": "flickr8k-images/2513260012_03d33305cf.jpg",
    "caption": [
      "A black dog is running after a white dog in the snow .",
      "Black dog chasing brown dog through snow",
      "Two dogs chase each other across the snowy ground .",
      "Two dogs play together in the snow .",
      "Two dogs running through a low lying body of water ."
    ]
  },
  {
    "image": "flickr8k-images/2903617548_d3e38d7f88.jpg",
    "caption": [
      "A little baby plays croquet .",
      "A little girl plays croquet next to a truck .",
      "The child is playing croquette by the truck .",
      "The kid is in front of a car with a put and a ball .",
      "The little boy is playing with a croquet hammer and ball beside the car ."
    ]
  },
]
```

2. 开始网络训练   
直接运行train.py即可开始训练。   

3. 训练结果预测   
训练结果预测需要用到两个文件，分别是clip.py和predict.py。我们首先需要去clip.py里面修改model_path。   
**model_path指向训练好的权值文件，在logs文件夹里。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
数据集参考提供的flickr8k数据集进行准备，包含图片文件和json文件。
\*.json为标注文件，\*.json的格式如下，image为图片的路径，caption为对应的文本，为一个列表，内容可以多条也可以单条：
```python
[
  {
    "image": "flickr8k-images/2513260012_03d33305cf.jpg",
    "caption": [
      "A black dog is running after a white dog in the snow .",
      "Black dog chasing brown dog through snow",
      "Two dogs chase each other across the snowy ground .",
      "Two dogs play together in the snow .",
      "Two dogs running through a low lying body of water ."
    ]
  },
  {
    "image": "flickr8k-images/2903617548_d3e38d7f88.jpg",
    "caption": [
      "A little baby plays croquet .",
      "A little girl plays croquet next to a truck .",
      "The child is playing croquette by the truck .",
      "The kid is in front of a car with a put and a ball .",
      "The little boy is playing with a croquet hammer and ball beside the car ."
    ]
  },
]
```

2. 开始网络训练  
直接运行train.py即可开始训练。
如果训练的是中文数据集，注意修改model_path与phi，使其对应。   

3. 训练结果预测   
训练结果预测需要用到两个文件，分别是clip.py和predict.py。我们首先需要去clip.py里面修改model_path。   
**model_path指向训练好的权值文件，在logs文件夹里。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py。   

### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在clip.py文件里面，在如下部分修改model_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。  
```python
_defaults = {
    #-------------------------------#
    #   指向logs文件夹下的权值文件
    #-------------------------------#
    "model_path"        : 'model_data/ViT-B-16-OpenAI.pth',
    #-------------------------------#
    #   模型的种类
    #   openai/VIT-B-16
    #   openai/VIT-B-16
    #   self-cn/VIT-B-32
    #-------------------------------#
    "phi"               : "openai/VIT-B-16",
    #--------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
    #   否则对图像进行CenterCrop
    #--------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py。

## 评估步骤 
1. 设置eval.py中的datasets_val_json_path和datasets_path。
2. 在clip.py里面修改model_path。**model_path指向训练好的权值文件，在logs文件夹里。**  
3. 运行eval.py即可获得评估结果。

### Reference
https://github.com/openai/CLIP   
https://github.com/alibaba/AliceMind  