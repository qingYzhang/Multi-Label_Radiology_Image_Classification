import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.ops.focal_loss as FocalLoss

import time
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report

from pipeline.dataset import DataSet


classes = ("Atelectasis","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis",
    "Effusion","Pneumonia","Pleural_thickening","Cardiomegaly","Nodule Mass","Hernia","No Finding")
def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="DenseNet121")  # ResNet101, DenseNet121, AlexNet
    # dataset
    parser.add_argument("--dataset", default="chest", type=str)
    parser.add_argument("--num_classes", default=14, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)   # ["randomflip", "ColorJitter", "resizedcrop", "RandAugment"]
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    # optimizer, default ADM
    parser.add_argument("--loss", default="BCE") #BCE, FOCAL
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight_decay")
    parser.add_argument("--total_epoch", default=30, type=int)
    # parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args
    
#data
if args.dataset == "chest":
    train_file = ["data/chest/train_data.json"]
    test_file = ['data/chest/test_data.json']
# train_aug = ["randomflip", "resizedcrop"]
# test_aug = []
# img_size = 448
# dataset = "chest"
# batch_size = 32
train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


# train_dataset = DataSet(train_file, train_aug, img_size, dataset)
# test_dataset = DataSet(test_file, test_aug, img_size, dataset)
# trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# BATCH_SIZE = 32
# lr = 1e-4
# weight_decay = 1e-5
# num_classes = 14


# model
if args.model == 'DenseNet121':
    model = models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, args.num_classes)
    model = model.cuda()
elif args.model == 'ResNet101':
    model = models.resnet101(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    model = model.cuda()
elif args.model == "AlexNet":
    model = models.alexnet(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, args.num_classes)
    model = model.cuda()

# define loss & optimizer
# logits = torch.randn(BATCH_SIZE, num_classes)
# targets = torch.randint(0, 2, (BATCH_SIZE, num_classes)).float()  # Example targets, shape: (batch_size, num_classes)
# criterion = F.binary_cross_entropy_with_logits(logit, target, reduction="mean")
if args.loss == "BCE":
    criterion = F.binary_cross_entropy_with_logits(reduction="mean")
elif args.loss == "FOCAL":
    criterion = FocalLoss(reduction="mean")

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




# best_val_auc = 0
# for epoch in range(args.total_epoch):
#     model.train()
#     epoch_begin = time.time()
#     for idx, data in enumerate(trainloader):
#         batch_begin = time.time() 
#         train_data = data['img'].cuda()
#         train_labels = data['target'].cuda()

#         optimizer.zero_grad()
#         y_pred = model(train_data)
#         if args.loss == "BCE":
#             loss = criterion(y_pred, train_labels)
#         elif args.loss == "FOCAL":
#             loss = criterion(y_pred, train_labels, reduction = "mean")
#         loss.backward()
#         optimizer.step()
#         t = time.time() - batch_begin
#         if index % args.print_freq == 0:
#             print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
#                 i, 
#                 args.batch_size * (index + 1),
#                 len(train_loader.dataset),
#                 loss,
#                 optimizer.param_groups[0]["lr"],
#                 float(t)
#             ))


# training
best_val_auc = 0
for epoch in range(args.total_epoch):
    for idx, data in enumerate(trainloader):
        train_data = data['img'].cuda()
        train_labels = data['target'].cuda()
        y_pred = model(train_data)
        if args.loss == "BCE":
            loss = criterion(y_pred, train_labels)
        elif args.loss == "FOCAL":
            loss = criterion(y_pred, train_labels, reduction = "mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # validation  
        if idx % args.print_freq == 0:
            model.eval()
            with torch.no_grad():    
                test_pred = []
                test_true = [] 
                for jdx, data in enumerate(testloader):
                    test_data = data['img'].cuda()
                    test_labels = data['target'].cuda()
                    y_pred = model(test_data)
                    test_pred.append(y_pred.cpu().detach().numpy())
                    test_true.append(test_labels.cpu().numpy())   
            
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                val_auc_mean =  roc_auc_score(test_true, test_pred) 


                print(test_true, test_pred)
                roc_auc_micro = roc_auc_score(test_true, test_pred, average='micro')
                roc_auc_macro = roc_auc_score(test_true, test_pred, average='macro')
                roc_auc_weighted = roc_auc_score(test_true, test_pred, average='weighted')
                roc_auc_samples = roc_auc_score(test_true, test_pred, average='samples')
                roc_auc_per_class = roc_auc_score(test_true, test_pred, average=None)

                print(f"AUC Micro: {roc_auc_micro:.4f}")
                print(f"AUC Macro: {roc_auc_macro:.4f}")
                print(f"AUC Weighted: {roc_auc_weighted:.4f}")
                print(f"AUC Samples: {roc_auc_samples:.4f}")
                print("AUC Per Class:")
                for class_name, auc_score in zip(classes, roc_auc_per_class):
                print(f"{class_name}: {auc_score:.4f}")

                model.train()

                if best_val_auc < val_auc_mean:
                    best_val_auc = val_auc_mean
                    torch.save(model.state_dict(), 'ce_pretrained_model.pth')

                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc ))
    # t = time.time() - epoch_begin
    # print("Epoch {} training ends, total {:.2f}s".format(i, t))