#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
from PIL import Image

from clip import Clip

if __name__ == "__main__":
    clip = Clip()
    
    image_path = "img/2090545563_a4e66ec76b.jpg"
    captions   = ["两个孩子在滑板上快乐地滑行。", "一位女士通过一个障碍的过程，而其他人都在后台。", "一只白色的狗正看着一只黑色的狗在一堆大石头旁边的草地上跳跃。", "一个户外溜冰场挤满了人。"]
    
    image = Image.open(image_path)
    clip.detect_image(image, captions)