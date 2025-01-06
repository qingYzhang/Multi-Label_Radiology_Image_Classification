import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import torchvision.models as models
from torchvision.ops import sigmoid_focal_loss as FocalLoss

import time
import timm
import argparse
import numpy as np
# from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report

from pipeline.dataset import DataSet


def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="ViT")  # ResNet101, DenseNet121, AlexNet, TResNet, ViT
    # dataset
    parser.add_argument("--dataset", default="ourdata", type=str) # chest, oai
    parser.add_argument("--num_classes", default=15, type=int) # 14 for chest, 5 for oai
    parser.add_argument("--train_aug", default=[], type=list)   # ['randomflip', 'randomrotate', 'randomperspective']
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--classes", default=("Atelectasis","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis",
    # "Effusion","Pneumonia","Pleural_thickening","Cardiomegaly","Nodule Mass","Hernia","No Finding"), type=tuple)
    # optimizer, default ADM
    parser.add_argument("--optimizer", default="Adam") # Adam, SGD
    parser.add_argument("--loss", default="FOCAL") # BCE, FOCAL
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float) # Specifically for SGD
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight_decay")
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args


def train(i, args, model, train_loader, optimizer):
    model.train()
    print()
    print("Train on Epoch {}".format(i))

    epoch_begin = time.time()
    for idx, data in enumerate(train_loader):
        batch_begin = time.time() 
        train_data = data['img'].cuda()
        train_labels = data['target'].cuda()
        # print(train_data.shape, train_data.size())
        optimizer.zero_grad()
        y_pred = model(train_data)

        # print(y_pred.shape, y_pred.size())

        if args.loss == "BCE":
            loss = F.binary_cross_entropy_with_logits(y_pred, train_labels, reduction='mean')
        elif args.loss == "FOCAL":
            loss = FocalLoss(y_pred, train_labels, reduction='mean')
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin
        if idx % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (idx + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))
    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader):
    model.eval()
    print("Test on Epoch {}".format(i))

    with torch.no_grad():    
        test_pred = []
        test_true = [] 
        for jdx, data in enumerate(test_loader):
            test_data = data['img'].cuda()
            test_labels = data['target'].cuda()

            y_pred = model(test_data)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_labels.cpu().numpy())   
        
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        print(test_true, test_pred)

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
        
        torch.save(model.state_dict(), f'../logs/epoch{i}_auc_{val_auc_mean:.2f}.pth')
        print ("Epoch {} testing ends, val_AUC = {:.4f}".format(i, val_auc_mean))


def main():
    args = Args()

    #data
    if args.dataset == "chest":
        train_file = ["data/chest/train_data.json"]
        test_file = ['data/chest/test_data.json']
        step_size = 4
    elif args.dataset == "ourdata":
        train_file = ["data/ourdata/train.json"]
        test_file = ['data/ourdata/test.json']
        step_size = 4
    elif args.dataset == "oai":
        train_file = ["data/oai/train_val_dataset.json"]
        test_file = ['data/oai/test_dataset.json']
        step_size = 4
    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    class_weights = train_dataset.class_weights
    # print(class_weights)
    sample_weights = np.array([class_weights[np.argmax(ann["target"])] for ann in train_dataset.anns])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=sampler)

    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

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
    elif args.model == "TResNet":
        print("Using TResNet.")
        model = timm.create_model('tresnet_l', pretrained=True)  # You can choose between 'tresnet_m', 'tresnet_l', 'tresnet_xl'
        model.head.fc = torch.nn.Linear(model.head.fc.in_features, args.num_classes)
    elif args.model == "ViT":
        print("Using ViT.")
        # Load a pretrained ViT model
        model = timm.create_model('vit_large_patch16_224', pretrained=True) 
        # Modify the classification head to match the number of classes
        model.head = torch.nn.Linear(model.head.in_features, args.num_classes)

    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # optimizer
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(
            [
                {'params': backbone, 'lr': args.lr},
                {'params': classifier, 'lr': args.lr * 10}
            ],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
        [
            {'params': backbone, 'lr': args.lr},
            {'params': classifier, 'lr': args.lr * 10}
        ],
        lr=args.lr,
        momentum=args.momentum,  # Add momentum if needed
        weight_decay=args.weight_decay
        )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # training and validation
    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer)
        val(i, args, model, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()