import pandas as pd
import os
import json
import PIL
from PIL import Image, ExifTags
from glob import glob
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import shutil
import numpy as np
from tqdm import tqdm
import yaml
import pickle as pkl
import config as CFG

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

s_path = CFG.detection_folder
datasets_path = "data/pills"

if not os.path.exists(datasets_path):
    os.mkdir(datasets_path)
    for x in ["base", "few", "all"]: 
        for s_ in ["_train", "_test"]:
            os.mkdir(os.path.join(datasets_path, x+s_))
            for s in ["images", "labels"]:
                os.mkdir(os.path.join(datasets_path, x+s_, s))
                
def multilabel_balance_split(img_ids, y, Label_names, shots=13):
    import random
    random.seed(20)
    img_ids = list(img_ids)
    ids_index = {id: i for i, id in enumerate(img_ids)}
    counter = {c: 0 for c in range(len(Label_names))}
    ids_test = []
    img_id_dict = {c: [id for id in img_ids if c in y[id]] for c in range(len(Label_names))}
    for cls in tqdm(range(len(Label_names))):
        if counter[cls] > shots or cls==len(Label_names)-1:
            continue
        while True:
            if len(img_id_dict[cls]) >= shots :
                sampled_ids = random.sample(img_id_dict[cls], shots)
            else:
                sampled_ids= img_id_dict[cls]
            for img in sampled_ids:
                if img in ids_test:
                    continue
                if len([c for c in y[img] if c == cls]) + counter[cls] > shots and any([counter[c] + len([k for k in y[img] if k == c])> shots-5 for c in range(len(Label_names))]):
                    continue
                # print("before:",counter[cls])
                ids_test.append(img)
                for c in y[img]:
                    counter[c] += 1
                if counter[cls] in [shots+i for i in range(-2, 17)]:
                    break
            # print(counter)
            # print(cls)
            # print(counter[cls])
            if counter[cls] in [shots+i for i in range(-2, 17)]:
                    break
    print(counter)
    index_test = [ids_index[id] for id in ids_test]
    print(len(index_test))
    return [idx for idx in ids_index.values() if idx not in index_test],index_test

def name2id(label_names):
    return {name: i for i, name in enumerate(label_names)}

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def main():
    dfs = []
    for path in os.listdir(s_path + "/json"):
        with open(s_path + "/json/" + path, "r") as f:
            annotation = json.load(f)
            img_path = annotation["path"]
            boxes = annotation["boxes"]
            if os.path.exists(s_path + "/images/" + img_path):
                img = Image.open(s_path + "/images/" + img_path)
            else:
                img = Image.open(s_path + "/images/" + img_path.split('.')[0] + '.JPG')
            w,h = exif_size(img)
            x_mid, y_mid, w_box, h_box, label = [], [], [], [], []
            for box in boxes: 
                x_mid.append((box["x"]/w + box["x"]/w + box["w"]/w)/2)
                y_mid.append((box["y"]/h + box["y"]/h + box["h"]/h)/2)
                w_box.append(box["w"]/w)
                h_box.append(box["h"]/h)

                label.append(box["label"])

            annotation_dict = {"image_id": path[:-5], "label":label, "x_mid":x_mid, "y_mid":y_mid, "w": w_box, "h": h_box}
            df = pd.DataFrame(annotation_dict)
            dfs.append(df)

    data_df = pd.concat(dfs, axis=0)
    
    Label_names = data_df["label"].unique()
    vl_ct = dict(data_df.label.value_counts())
    # print(f'Value count: {vl_ct}')
    base_names = [name for name in Label_names if vl_ct[name] >= 100]
    base_names = sorted(base_names)
    #Few shot pills are the ones with less than 100 samples and greater than 30 samples
    few_shot_names = [name for name in Label_names if 30 < vl_ct[name] < 100]
    eliminated_names = [name for name in Label_names if vl_ct[name] <= 30]
    # print(f'eliminate class: {eliminated_names}')
    eliminated_ids = data_df[data_df["label"].isin(eliminated_names)].image_id.unique()
    data_df = data_df[data_df.image_id.isin([x for x in data_df.image_id.unique() if x not in eliminated_ids])]
    # few_shot_names = sorted([name for name in Label_names if name not in base_names])
    few_shot_names = sorted(few_shot_names)
    # print(f'few shot name: {few_shot_names}')
    Label_names_elim = base_names + few_shot_names
    ids = name2id(sorted(Label_names_elim))

    with open(datasets_path + "/name2id.pkl", "wb") as f:
        pkl.dump(ids, f)
    with open(datasets_path + "/base_names.pkl", "wb") as f:
        pkl.dump(base_names, f)
    with open(datasets_path + "/few_shot_names.pkl", "wb") as f:
        pkl.dump(few_shot_names, f)
    
    print("We have number of base classes:", len(base_names))
    print("We have number of few shot classes:", len(few_shot_names))
    imgs_id = data_df.image_id.unique()
    #Images that contain few shot classes (possibly contain base classes)
    imgs_id_few_neg = data_df[data_df.label.isin(few_shot_names)].image_id.unique()
    #Image that only contain base classes
    base_df = data_df[data_df.image_id.isin([i for i in imgs_id if i not in imgs_id_few_neg])]
    #Image that only contain base classes (possibly contain few shot classes)
    imgs_id_base_neg = data_df[data_df.label.isin(base_names)].image_id.unique()
    #Image that only contain few shot classes
    few_df = data_df[data_df.image_id.isin([i for i in imgs_id if i not in imgs_id_base_neg])]
    #Image that contain both few-shot and base classes
    imgs_id_all = [x for x in imgs_id_base_neg if x in imgs_id_few_neg]
    all_df = data_df[data_df.image_id.isin([i for i in imgs_id if i in imgs_id_all])]
    # print(all_df[all_df.label == "Aerius_5mg"].shape)

    cont_base_id_map = {k:i for i,k in enumerate([ids[x] for x in base_names])}
    cont_all_id_map = {k:i for i,k in enumerate([ids[x] for x in Label_names_elim])}
    cont_few_id_map = {k:i for i,k in enumerate([ids[x] for x in few_shot_names])}
    cont_data_id_map = {k:i for i,k in enumerate([ids[x] for x in Label_names_elim])}
    print(cont_data_id_map)
    
    base_df["label_id"] = np.zeros(len(base_df))
    base_df["label_id"] = base_df.label.map(ids)
    few_df["label_id"] = np.zeros(len(few_df))
    few_df["label_id"] = few_df.label.map(ids)
    all_df["label_id"] = np.zeros(len(all_df))
    all_df["label_id"] = all_df.label.map(ids)
    data_df["label_id"] = np.zeros(len(data_df))
    data_df["label_id"] = data_df.label.map(ids)
    
    # imgs_id_base = base_df.image_id.unique()
    # imgs_id_all = all_df.image_id.unique()
    # imgs_id_few = few_df.image_id.unique()
    # y_base = []
    # for image_id in tqdm(imgs_id_base):
    #     lt = np.zeros(len(base_names))
    #     df = list(base_df[base_df['image_id'] == image_id]['label_id'])
    #     df = [cont_base_id_map[int(x)] for x in df]
    #     lt[df] = 1
    #     y_base.append(list(lt))
    
    # msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # train_base_index,  test_base_index= [(x, y) for x,y in msss1.split(imgs_id_base, y_base)][0]
    # train_files = imgs_id_base[train_base_index]
    # print(base_df.head())
    # train_dts = base_df[base_df.image_id.isin([i for i in train_files])]
    # print(train_dts.head())
    # vlt_cnt_train = dict(train_dts.label_id.value_counts())
    # print(vlt_cnt_train)
    
    # test_files = imgs_id_base[test_base_index]
    # test_dts = base_df[base_df.image_id.isin([i for i in test_files])]
    # vlt_cnt_test = dict(test_dts.label_id.value_counts())
    # print(vlt_cnt_test)
    # for image_id in list(set(train_files)):
    #     shutil.copy(s_path + '/images/' + image_id + '.jpg', os.path.join(datasets_path, "base_train", "images", image_id + ".jpg"))
    #     shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "base_train", "labels", image_id + ".json"))
    # for image_id in list(set(test_files)):
    #     shutil.copy(s_path + '/images/' + image_id + '.jpg', os.path.join(datasets_path, "base_test", "images", image_id + ".jpg"))
    #     shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "base_test", "labels", image_id + ".json"))

    y_data = []
    for image_id in tqdm(imgs_id):
        lt = np.zeros(len(Label_names_elim))
        df = list(data_df[data_df['image_id'] == image_id]['label_id'])
        df = [cont_data_id_map[int(x)] for x in df]
        lt[df] = 1
        y_data.append(list(lt))
    
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_data_index,  test_data_index= [(x, y) for x,y in msss1.split(imgs_id, y_data)][0]
    train_files = imgs_id[train_data_index]
    test_files = imgs_id[test_data_index]
    # train_dts = data_df[data_df.image_id.isin([i for i in train_files])]
    # print(train_dts.head())
    # vlt_cnt_train = dict(train_dts.label_id.value_counts())
    # print(vlt_cnt_train)
    # for i in range(len(Label_names_elim)):
    #     if i not in vlt_cnt_train.keys():
    #         print(i)

    # test_dts = data_df[data_df.image_id.isin([i for i in test_files])]
    # vlt_cnt_test = dict(test_dts.label_id.value_counts())
    # print(vlt_cnt_test)
    # for i in range(len(Label_names_elim)):
    #     if i not in vlt_cnt_test.keys():
    #         print(i)
    for image_id in list(set(train_files)):
        shutil.copy(s_path + '/images/' + image_id + '.jpg', os.path.join(datasets_path, "data_train", "images", image_id + ".jpg"))
        shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "data_train", "labels", image_id + ".json"))
    for image_id in list(set(test_files)):
        shutil.copy(s_path + '/images/' + image_id + '.jpg', os.path.join(datasets_path, "data_test", "images", image_id + ".jpg"))
        shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "data_test", "labels", image_id + ".json"))

    # if len(imgs_id_few)>10:
    #     y_few = []
    #     for image_id in tqdm(imgs_id_few):
    #         lt = np.zeros(len(few_shot_names))
    #         df = list(few_df[few_df['image_id'] == image_id]['label_id'])
    #         df = [cont_few_id_map[int(x)] for x in df]
    #         lt[df] = 1
    #         y_few.append(list(lt))
        
    #     msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    #     train_few_index,  test_few_index= [(x, y) for x,y in msss2.split(imgs_id_few, y_few)][0]
    #     train_files = imgs_id_few[train_few_index]
    #     test_files = imgs_id_few[test_few_index]
    #     for image_id in list(set(train_files)):
    #         shutil.copy(s_path + '/pics/' + image_id + '.jpg', os.path.join(datasets_path, "few_train", "images", image_id + ".jpg"))
    #         shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "few_train", "labels", image_id + ".json"))
    #     for image_id in list(set(test_files)):
    #         shutil.copy(s_path + '/pics/' + image_id + '.jpg', os.path.join(datasets_path, "few_test", "images", image_id + ".jpg"))
    #         shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "few_test", "labels", image_id + ".json"))

    # y_all = []
    # for image_id in tqdm(imgs_id_all):
    #     lt = np.zeros(len(imgs_id_all))
    #     # print(all_df[all_df['image_id'] == image_id])
    #     df = list(all_df[all_df['image_id'] == image_id]['label_id'])
    #     df = [cont_all_id_map[int(x)] for x in df]
    #     lt[df] = 1
    #     y_all.append(list(lt))
    
    # msss3 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # train_all_index,  test_all_index= [(x, y) for x,y in msss3.split(imgs_id_all, y_all)][0]
    # train_files = imgs_id_all[train_all_index]
    # # train_dts = all_df[all_df['image_id'] in train_files]
    # # vlt_cnt_train = dict(train_dts.label_id.value_counts())
    # # print(vlt_cnt_train)
    
    # test_files = imgs_id_all[test_all_index]
    # # test_dts = all_df[all_df['image_id'] in test_files]
    # # vlt_cnt_test = dict(test_dts.label_id.value_counts())
    # # print(vlt_cnt_test)

    # for image_id in list(set(train_files)):
    #     shutil.copy(s_path + '/images/' + image_id + '.jpg', os.path.join(datasets_path, "all_train", "images", image_id + ".jpg"))
    #     shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "all_train", "labels", image_id + ".json"))
    # for image_id in list(set(test_files)):
    #     shutil.copy(s_path + '/images/' + image_id + '.jpg', os.path.join(datasets_path, "all_test", "images", image_id + ".jpg"))
    #     shutil.copy(s_path + '/json/' + image_id + '.json', os.path.join(datasets_path, "all_test", "labels", image_id + ".json"))

if __name__ == '__main__':
    if not os.path.exists(datasets_path):
        os.mkdir(datasets_path)
    for x in ["base", "few", "all", "data"]: 
        for s_ in ["_train", "_test"]:
            os.mkdir(os.path.join(datasets_path, x+s_))
            for s in ["images", "labels"]:
                os.mkdir(os.path.join(datasets_path, x+s_, s))
    main()