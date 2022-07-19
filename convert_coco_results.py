from PIL import ImageFile, Image
import os
import pandas as pd
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

def test_img_loader():
    root_path = 'data/pills/data_train_ai4vn/pill/image'
    # for img_f in os.listdir('data/pills/data_train_ai4vn/pill/image'):
    img_f = 'VAIPE_P_449_9.jpg'
    print(img_f)
    img = Image.open(os.path.join(root_path, img_f))
    img.load()
    print(img)

def convert_coco_to_csv(path, save_path):
    df = pd.DataFrame(columns=['image_id', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    coco_res = json.load(open(path, 'r'))
    for ann in coco_res:
        row = {}
        row['image_name'] = ann['image_id']
        row['class_id'] = ann['category_id']
        row['confidence_score'] = ann['score']
        row['x_min'] = ann['bbox'][0]
        row['y_min'] = ann['bbox'][1]
        row['x_max'] = ann['bbox'][0] + ann['bbox'][2]
        row['y_max'] = ann['bbox'][1] + ann['bbox'][3]
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
    
    df.to_csv(save_path + 'results.csv', index=False)


if __name__ == '__main__':
    convert_coco_to_csv('logs/KGP_e2e_gtn_ai4vn/inference/coco_instances_results.json', 
                        'logs/KGP_e2e_gtn_ai4vn/inference/')

