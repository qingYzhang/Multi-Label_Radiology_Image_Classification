import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report


# from libauc.losses.auc import AUCMLoss
from pipeline.losses import CrossEntropyLoss
from pipeline.adam import Adam
from pipeline.densenet import densenet121 as DenseNet121
from pipeline.resnet import resnet101 as ResNet101
from pipeline.dataset import DataSet

classes = ("Atelectasis","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis",
    "Effusion","Pneumonia","Pleural_thickening","Cardiomegaly","Nodule Mass","Hernia","No Finding")
def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="DenseNet121")
    # parser.add_argument("--num_heads", default=1, type=int)
    # parser.add_argument("--lam",default=0.1, type=float)
    # parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="chest", type=str)
    # parser.add_argument("--num_cls", default=14, type=int)
    # parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    # parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--BATCH_SIZE", default=32, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=1e-4, type=float)
    # parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight_decay")
    # parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    # parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args
    
#data
# dataloader
# root = '../CheXpert/CheXpert-v1.0/'
# Index: -1 denotes multi-label mode including 5 diseases
train_file = ["data/train_data.json"]
test_file = ['data/test_data.json']
train_aug = ["randomflip", "resizedcrop"]
test_aug = []
img_size = 448
dataset = "chest"
batch_size = 32
# train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
# test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


train_dataset = DataSet(train_file, train_aug, img_size, dataset)
test_dataset = DataSet(test_file, test_aug, img_size, dataset)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# traindSet = CheXpert('../CheXpert/train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
# testSet =  CheXpert('../CheXpert/valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1)
# trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
# testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)

BATCH_SIZE = 32
lr = 1e-4
weight_decay = 1e-5

# model
model = 'ResNet101'
if model == 'DenseNet121':
    model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=14)
    model = model.cuda()
elif model == 'ResNet101':
    model = ResNet101(pretrained=False, progress=True, activations='relu', num_classes=14)

# define loss & optimizer
CELoss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training
best_val_auc = 0
for epoch in range(30):
    for idx, data in enumerate(trainloader):
    #   print(idx, "..............................................",data)
      train_data = data['img'].cuda()
      train_labels = data['target'].cuda()
      #   train_data, train_labels = data
      #   train_data, train_labels  = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      loss = CELoss(y_pred, train_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
      # validation  
      if idx % 400 == 0:
         model.eval()
         with torch.no_grad():    
              test_pred = []
              test_true = [] 
              for jdx, data in enumerate(testloader):
                  test_data = data['img'].cuda()
                  test_labels = data['target'].cuda()
                  #   test_data, test_labels = data
                  #   test_data = test_data.cuda()
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
