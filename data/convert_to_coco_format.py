import json 
import os
import pickle as pkl
import cv2
import tqdm

with open("data/name2id.pkl", "rb") as f:
    name2id = pkl.load(f)
    
id_box = 0
for img_set in ["train", "test"]:
    converted_vaipe = {"images": [], "annotations": [], "categories": []}
    anno_dir = "data/all_{}/labels".format(img_set)
    img_dir = "data/all_{}/images/".format(img_set)
    for anno_json in tqdm.tqdm(os.listdir(anno_dir)):
            image_id = anno_json.split(".")[0]
            with open(os.path.join(anno_dir, anno_json)) as f:
                annos = json.load(f)
                for box in annos["boxes"]:
                    anno = {"segmentation": [[]], 
                            "bbox": [box["x"], box["y"], box["w"], box["h"]], 
                            "category_id": name2id[box["label"]],
                            "image_id": image_id,
                            "id": id_box,
                            "area": box["w"]*box["h"]
                    }
                    id_box +=1
                    converted_vaipe["annotations"].append(anno)
                
            img = {
                "file_name": "data/all_{}/images/{}.jpg".format(img_set,image_id)
            }
            height, width = cv2.imread(img["file_name"]).shape[:2]
            img["height"], img["width"] = height, width
            img["id"] = image_id
            converted_vaipe["images"].append(img)
    for name in name2id.keys():
        converted_vaipe["categories"].append({"id": name2id[name], "name": name})
    with open("data/all_{}/instances_{}.json".format(img_set, img_set), "w") as f:
        json.dump(converted_vaipe, f)
