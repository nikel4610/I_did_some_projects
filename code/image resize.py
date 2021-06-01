import glob
from PIL import Image
imglist = glob.glob("D:/project/untitled1/sampleimages/*.jpg") #경로

img = Image.open(imglist[0]) 

for img_path in imglist: #덮어씀
    img = Image.open(img_path)
    img.resize((113, 48)).save(img_path) #크기 지정
