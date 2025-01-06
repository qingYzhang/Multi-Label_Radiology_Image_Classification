import torch
from torch.utils.data import DataLoader

import torchvision.models as models

import csv
import json
import time
import pydicom
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import v2 as transforms
from sklearn.metrics import roc_auc_score, classification_report

from pipeline.dataset import DataSet



def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="DenseNet121")  # ResNet101, DenseNet121, AlexNet
    # dataset
    parser.add_argument("--dataset", default="ourdata", type=str) # chest, oai
    parser.add_argument("--num_classes", default=15, type=int) # 14 for chest, 5 for oai
    parser.add_argument("--test_aug", default=[], type=list) # ['randomflip', 'randomrotate', 'randomperspective']
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--load_from", default="../logs/epoch22_auc_0.57.pth", type=str)

    args = parser.parse_args()
    return args
    

def val(args, model, test_loader):
    model.eval()
    print("Test on Pretrained Model")


    with open('predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['accession', 'y_pred'])  # Write the header

        with torch.no_grad():    
            test_pred = []
            test_true = [] 
            for jdx, data in enumerate(test_loader):
           
                accession = data['accession']
                test_data = data['img'].cuda()
                test_labels = data['target'].cuda()

                y_pred = model(test_data)
                print(y_pred)

                y_pred_min = y_pred.min(dim=1, keepdim=True).values
                y_pred_max = y_pred.max(dim=1, keepdim=True).values
                y_pred_normalized = (y_pred - y_pred_min) / (y_pred_max - y_pred_min + 1e-8)  # Add epsilon to avoid division by zero

                print(y_pred_normalized)
                y_pred_thresholded = (y_pred_normalized > 0.9).float()
                print(y_pred_thresholded)
                test_pred.append(y_pred.cpu().detach().numpy())
                test_true.append(test_labels.cpu().numpy())   

                for acc, pred in zip(accession, y_pred_thresholded.cpu().numpy()):
                    writer.writerow([acc] + pred.tolist())  # Write the accession and all label predictions

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
    # with open("./data/ourdata/newtest.json", "r") as file:
    #     data = json.load(file)


    # with open('predictions.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['accession', 'y_pred'])  # Write the header
    #     # Add accession number to each entry
    #     for item in data:
    #         accession = item["accession"]
    #         img_path = item["img_path"]
    #         dicom = pydicom.dcmread(img_path)
            
    #         # study_description = dataset.get((0x0008, 0x1030), 'Unknown View')
    #         img_data = dicom.pixel_array
    #         img = Image.fromarray(img_data)
    #         img = img.convert("RGB")

    #         t= []
            
    #         t.append(transforms.Resize((448, 448)))

    #         augment = transforms.Compose(t)
    #         img = augment(img)
    #         transform = transforms.Compose([
             
    #             transforms.ToTensor(),          # Convert image to tensor
    #             transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalize the image
    #         ])

    #         # Apply the transformation to the image
 
    #         img = transform(img).to("cuda")

    #         print(img)
    #         y_pred = model(img)


    #         for acc, pred in zip(accession.cpu().numpy(), y_pred.cpu().numpy()):
    #                 writer.writerow(['accession'] + [f'y_pred_{i}' for i in range(model.output_size)])  # Write header with multiple labels



    

    if args.dataset == "chest":
        train_file = ["data/chest/train_data.json"]
        test_file = ['data/chest/test_data.json']
        step_size = 4
    elif args.dataset == "oai":
        train_file = ["data/oai/train_val_dataset.json"]
        test_file = ['data/oai/test_dataset.json']
        step_size = 4
    elif args.dataset == "ourdata":
        train_file = ["data/ourdata/train.json"]
        test_file = ['data/ourdata/newtest.json']
        step_size = 4

    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val(args, model, test_loader)

if __name__ == "__main__":
    main()
