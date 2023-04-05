from PIL import Image

from clip import CLIP

if __name__ == "__main__":
    clip = CLIP()
    
    # 图片的路径
    image_path = "img/2090545563_a4e66ec76b.jpg"
    # 寻找对应的文本，4选1
    captions   = [
        "The two children glided happily on the skateboard.", 
        "A woman walks through a barrier while everyone else is backstage.", 
        "A white dog was watching a black dog jump on the grass next to a pile of big stones.", 
        "An outdoor skating rink was crowded with people."
    ]
    
    image = Image.open(image_path)
    probs = clip.detect_image(image, captions)
    print("Label probs:", probs)