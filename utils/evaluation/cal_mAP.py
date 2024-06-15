import os
import numpy as np
import torch
import json


def json_map(cls_id, pred_json, ann_json, types):
    assert len(ann_json) == len(pred_json)
    num = len(ann_json)
    predict = np.zeros((num), dtype=np.float64)
    target = np.zeros((num), dtype=np.float64)

    for i in range(num):
        predict[i] = pred_json[i]["scores"][cls_id]
        target[i] = ann_json[i]["target"][cls_id]

    if types == 'wider':
        tmp = np.where(target != 99)[0]
        predict = predict[tmp]
        target = target[tmp]
        num = len(tmp)

    if types == 'voc07' or types == 'chest':
        tmp = np.where(target != 0)[0]
        # print("tmp...",tmp)
        predict = predict[tmp]
        # print("pre...",predict)
        target = target[tmp]
        # print("tar...",target)
        neg_id = np.where(target == -1)[0]
        # print("ned_id...", neg_id)
        target[neg_id] = 0
        num = len(tmp)
        # print("num...",num)


    tmp = np.argsort(-predict)
    # print("tmp...",tmp)
    target = target[tmp]
    # print("tar...",target)
    predict = predict[tmp]
    # print("pre...",predict)

    pre, obj = 0, 0
    for i in range(num):
        if target[i] == 1:
            # print("flag")
            obj += 1.0
            # print(obj)
            pre += obj / (i+1)
    # print(pre,obj,obj==0)
    pre /= obj
    # if obj > 0:
    #     pre /= obj
    # else:
    #     pre = 0 
    return pre













