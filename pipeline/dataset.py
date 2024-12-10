import json
import numpy as np
from PIL import Image

import torch
import pydicom
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class DataSet(Dataset):
    def __init__(self,
                ann_files,
                augs,
                img_size,
                dataset,
                ):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ] 
            # We normalize the image data to [0, 1]
            # Or 'ImageNet' Normalization method
        )
        self.anns = []
        self.load_anns()
        self.class_weights = self.compute_class_weights()
        # print(self.augment)

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'randomrotate' in augs:
            t.append(transforms.RandomRotation(degrees=(0, 180)))
        if 'randomperspective' in augs:
            t.append(transforms.RandomPerspective())
    
        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)
    
    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def compute_class_weights(self):
        targets = [ann["target"] for ann in self.anns]
        num_classes = len(targets[0])
        class_counts = np.zeros(num_classes)
        for target in targets:
            class_counts += np.array(target)
        class_weights = 1.0 / class_counts
        return class_weights

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]


        # make sure to change this when you use dcm
        dicom = pydicom.dcmread(ann["img_path"])
        
        # study_description = dataset.get((0x0008, 0x1030), 'Unknown View')
        img_data = dicom.pixel_array
        img = Image.fromarray(img_data)
        img = img.convert("RGB")


        # this is for png
        # img = Image.open(ann["img_path"]).convert("RGB")
        img = self.augment(img)
        img = self.transform(img)
        message = {
            "img_path": ann["img_path"],
            "target": torch.Tensor(ann["target"]),
            "img": img
        }

        return message
        # finally, if we use dataloader to get the data, we will get
        # {
        #     "img_path": list, # length = batch_size
        #     "target": Tensor, # shape: batch_size * num_classes
        #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
        # }