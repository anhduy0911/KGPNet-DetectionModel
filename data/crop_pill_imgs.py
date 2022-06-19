import glob
import cv2
import json
import os
from tqdm import tqdm
import pandas as pd
import config as CFG

if not os.path.exists("data/pills/pill_cropped"):
    os.mkdir("data/pills/pill_cropped")

# Thanh update
# json_mapping = None
# try:
#     json_mapping = pd.read_csv("/home/aiotlabws/Workspace/Project/hieunm/emed/statistic/results/data_uong_thuoc/json_mapping.csv")
#     print("Read mapping CSV file succeed !")
# except:
#     print("ERROR: Read mapping CSV file failed !")
#     exit()

# json_mapping_dict = json_mapping.set_index('filename')['id'].to_dict()
# Thanh update

error_list = []

for path_json in tqdm(glob.glob(CFG.detection_folder + "json/*.json")):
    file_json = open(path_json)
    data_json = json.load(file_json)
    path_img = data_json["path"]
    boxes = data_json["boxes"]
    img = cv2.imread(CFG.detection_folder + "images/" + path_img)
    # idx = json_mapping_dict[path_img]
    idx = path_img
    pres_idx = str(idx.split(".")[0])
    os.mkdir(f"data/pills/pill_cropped/{pres_idx}/")
    try:
        for i, box in enumerate(boxes):
            x = box["x"]
            y = box["y"]
            w = box["w"]
            h = box["h"]
            label = box["label"]

            crop_img = img[y:y+h, x:x+w]
            path_folder_img = f"data/pills/pill_cropped/{pres_idx}/" + label

            # File_name
            file_name = path_folder_img + "/" + \
                pres_idx + '-' + str(i) + ".jpg"

            # Create new folder if not exists
            if not os.path.exists(path_folder_img):
                os.mkdir(path_folder_img)

            # If exist file name
#             if os.path.exists(file_name):
#                 file_name = file_name.split(".jpg")[0] + "_" + str(len(glob.glob1(path_folder_img,"{}*".format(path_img.split(".")[0])))) + ".jpg"
            cv2.imwrite(file_name, crop_img)
    except:
        img_error = str(idx) + '-' + str(i) + ".jpg"
        error_list.append(img_error)
        print("Error: ", img_error)

dict = {'error': error_list}
df = pd.DataFrame(dict)
df.to_csv("data/pills/error_list.csv")
