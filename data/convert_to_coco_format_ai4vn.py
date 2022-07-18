import json 
import os
import pickle as pkl
import cv2
import tqdm

def convert_to_coco():
    id_box = 0
    for img_set in ["train_ai4vn", "test_ai4vn"]:
        converted_vaipe = {"images": [], "annotations": [], "categories": []}
        anno_dir = "data/pills/data_{}/pill/label/".format(img_set)
        img_dir = "data/pills/data_{}/pill/image/".format(img_set)
        for anno_json in tqdm.tqdm(os.listdir(anno_dir)):
                image_id = anno_json.split(".")[0]
                with open(os.path.join(anno_dir, anno_json)) as f:
                    annos = json.load(f)
                    for box in annos:
                        anno = {
                                "bbox": [box["x"], box["y"], box["w"], box["h"]], 
                                "bbox_mode": 1,
                                "category_id": box["label"],
                                "image_id": image_id,
                                "id": id_box,
                                "area": box["w"]*box["h"],
                                "iscrowd": 0
                        }
                        id_box +=1
                        converted_vaipe["annotations"].append(anno)
                    
                img = {
                    "file_name": "data/pills/data_{}/pill/image/{}.jpg".format(img_set,image_id)
                }
                height, width = cv2.imread(img["file_name"]).shape[:2]
                img["height"], img["width"] = height, width
                img["id"] = image_id
                converted_vaipe["images"].append(img)
        for id in range(108):
            converted_vaipe["categories"].append({"id": id, "name": str(id)})
        
        with open("data/pills/data_{}/instances_{}.json".format(img_set, img_set), "w") as f:
            json.dump(converted_vaipe, f)

if __name__ == '__main__':
    # convert_to_coco()

    # visualize for testing
    dataset_dicts = json.load(open("data/pills/data_train_ai4vn/instances_train_ai4vn.json"))

    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    import matplotlib.pyplot as plt
    # register 2 datasets
    register_coco_instances("pills_train_ai4vn", {}, "data/pills/data_train_ai4vn/instances_train_ai4vn.json", "")
    register_coco_instances("pills_test_ai4vn", {}, "data/pills/data_test_ai4vn/instances_test_ai4vn.json", "")

    pill_metadata = MetadataCatalog.get("pills_train_ai4vn")
    d = dataset_dicts["images"][5]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=pill_metadata)
    d_annot = [i for i in dataset_dicts["annotations"] if i["image_id"] == d["id"]]
    
    out = visualizer.draw_dataset_dict({"annotations": d_annot})

    plt.imshow(out.get_image())
    plt.savefig('test.png', dpi=200)