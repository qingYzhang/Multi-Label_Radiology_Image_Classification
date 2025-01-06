import torch
from torch.utils.data import DataLoader

import torchvision.models as models

import time
import argparse
import numpy as np
# from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report

from pipeline.dataset import DataSet



def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="ResNet101")  # ResNet101, DenseNet121, AlexNet
    # dataset
    parser.add_argument("--dataset", default="oai", type=str) # chest, oai
    parser.add_argument("--num_classes", default=5, type=int) # 14 for chest, 5 for oai
    parser.add_argument("--test_aug", default=[], type=list) # ['randomflip', 'randomrotate', 'randomperspective']
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--load_from", default="../logs/epoch5_auc_0.88.pth", type=str)

    args = parser.parse_args()
    return args
    

def val(args, model, test_loader):
    model.eval()
    print("Test on Pretrained Model")

    with torch.no_grad():    
        test_pred = []
        test_true = [] 
        for jdx, data in enumerate(test_loader):
            test_data = data['img'].cuda()
            test_labels = data['target'].cuda()
            print(test_data.shape)

            y_pred = model(test_data)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_labels.cpu().numpy())   
        
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        val_auc_mean =  roc_auc_score(test_true, test_pred)
        
        # roc_auc_micro = roc_auc_score(test_true, test_pred, average='micro')
        # roc_auc_macro = roc_auc_score(test_true, test_pred, average='macro')
        # roc_auc_weighted = roc_auc_score(test_true, test_pred, average='weighted')
        # roc_auc_samples = roc_auc_score(test_true, test_pred, average='samples')
        # roc_auc_per_class = roc_auc_score(test_true, test_pred, average=None)

        # print(f"AUC Micro: {roc_auc_micro:.4f}")
        # print(f"AUC Macro: {roc_auc_macro:.4f}")
        # print(f"AUC Weighted: {roc_auc_weighted:.4f}")
        # print(f"AUC Samples: {roc_auc_samples:.4f}")
        # print("AUC Per Class:")
        # for class_name, auc_score in zip(args.classes, roc_auc_per_class):
        #     print(f"{class_name}: {auc_score:.4f}")
        
        torch.save(model.state_dict(), f'../logs/test_auc_{val_auc_mean:.2f}.pth')
        print ("Testing ends, val_AUC = {:.4f}".format(val_auc_mean))

def main():
    args = Args()

    # model
    if args.model == 'DenseNet121':
        model = models.densenet121(pretrained=True)
        print("Using DenseNet121.")
        model.classifier = torch.nn.Linear(model.classifier.in_features, args.num_classes)
    elif args.model == 'ResNet101':
        model = models.resnet101(pretrained=True)
        print("Using ResNet101.")
        model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "AlexNet":
        print("Using AlexNet.")
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, args.num_classes)
    model.cuda()

    print("Loading weights from {}".format(args.load_from))
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.module.load_state_dict(torch.load(args.load_from))
    else:
        model.load_state_dict(torch.load(args.load_from))

    # data
    if args.dataset == "chest":
        train_file = ["data/chest/train_data.json"]
        test_file = ['data/chest/test_data.json']
        step_size = 4
    elif args.dataset == "oai":
        train_file = ["data/oai/train_val_dataset.json"]
        test_file = ['data/oai/test_dataset.json']
        step_size = 4

    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val(args, model, test_loader)

if __name__ == "__main__":
    main()
