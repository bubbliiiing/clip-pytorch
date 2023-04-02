#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
from PIL import Image

from clip import CLIP

if __name__ == "__main__":
    clip = Clip()
    
    image_path = "img/2090545563_a4e66ec76b.jpg"
    captions   = [
        "The two children glided happily on the skateboard.", 
        "A woman walks through a barrier while everyone else is backstage.", 
        "A white dog was watching a black dog jump on the grass next to a pile of big stones.", 
        "An outdoor skating rink was crowded with people."
    ]
    
    image = Image.open(image_path)
    probs = clip.detect_image(image, captions)
    print("Label probs:", probs)