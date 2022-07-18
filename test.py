from PIL import ImageFile, Image
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

root_path = 'data/pills/data_train_ai4vn/pill/image'
# for img_f in os.listdir('data/pills/data_train_ai4vn/pill/image'):
img_f = 'VAIPE_P_449_9.jpg'
print(img_f)
img = Image.open(os.path.join(root_path, img_f))
img.load()
print(img)

