import PIL
from PIL import Image

folder_path='./Data/Black_White/' 
img='gray_8n.jpg'
img=folder_path+img

width, height = PIL.Image.open(img).size
print(width, height)