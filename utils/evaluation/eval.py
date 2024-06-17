import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import classification_report

from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3

chest_classes = ("Atelectasis","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis",
    "Effusion","Pneumonia","Pleural_thickening","Cardiomegaly","Nodule Mass","Hernia","No Finding")
voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")
coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
wider_classes = (
                "Male","longHair","sunglass","Hat","Tshiirt","longSleeve","formal",
                "shorts","jeans","longPants","skirt","faceMask", "logo","stripe")

class_dict = {
    "chest": chest_classes,
    "voc07": voc_classes,
    "coco": coco_classes,
    "wider": wider_classes,
}



def evaluation(result, types, ann_path):
    print("Evaluation")
    classes = class_dict[types]
    ann_json = json.load(open(ann_path, "r"))

    aps = np.zeros(len(classes), dtype=np.float64)
    pred_json = result
    # print(pred_json)
    for i, _ in enumerate(tqdm(classes)):
        ap = json_map(i, pred_json, ann_json, types)
        aps[i] = ap
    OP, OR, OF1, CP, CR, CF1 = json_metric(pred_json, ann_json, len(classes), types)
    print("mAP: {:4f}".format(np.mean(aps)))
    print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
    print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))



    y_true = []
    y_pred = []

    # Convert annotations and predictions to lists of true and predicted labels
    for ann, pred in zip(ann_json, result):
        true_labels = ann['target']
        true_labels = [0 if score == -1 else 1 for score in true_labels]
        pred_scores = pred['scores']
        pred_labels = [1 if score >= 0.5 else 0 for score in pred_scores]  # Assuming a threshold of 0.5
        
        y_true.append(true_labels)
        y_pred.append(pred_labels)

    # Flatten the lists for the classification report
    # y_true_flat = [label for sublist in y_true for label in sublist if label != -1]
    # y_pred_flat = [label for sublist in y_pred for label in sublist if label != -1]

    # Generate the classification report
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)