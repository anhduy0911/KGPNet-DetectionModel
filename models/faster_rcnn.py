from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
import config as CFG
import os
import torch
import json

def train(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = CFG.base_log + args.name
    cfg.DATASETS.TRAIN = ("pills_train",)
    cfg.DATASETS.TEST = ("pills_test",)
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.KEYPOINT_ON = False
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    # save_model(trainer, args.name)
    trainer.train()
    # save_model(trainer, args.name)

def save_model(trainer, name):
    state_dict = trainer.model.state_dict()
    torch.save(state_dict, os.path.join(trainer.cfg.OUTPUT_DIR, name + ".pth"))

def test(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = CFG.base_log + args.name
    cfg.DATASETS.TEST = ("pills_test",)
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.KEYPOINT_ON = False

    predictor = DefaultPredictor(cfg)

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.data import DatasetCatalog
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import matplotlib.pyplot as plt

    # visualization
    test_dict = DatasetCatalog.get("pills_test")
    d = test_dict[2]
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(d['annotations'])
    v = Visualizer(im[:, :, ::-1],
                    metadata=d,
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image())
    plt.savefig('eval.png', dpi=300)
    
    # # evaluation
    evaluator = COCOEvaluator("pills_test", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "pills_test")
    result = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(result)
    with open(cfg.OUTPUT_DIR + "/results.json", "w") as f:
        json.dump(result, f)
    