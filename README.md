## Requirements
- Python 3.7
- pytorch 1.6
- torchvision 0.7.0
- pycocotools 2.0
- tqdm 4.49.0, pillow 7.2.0

## Dataset
We expect Chest dataset to have the following structure:
```
Dataset/
|--Chest
|----all_images
|----Data_Entry_2017.csv
|----train_val_list.txt
|----test_list.txt
...
```
## Generate json file (for implementation) of these datasets.
```shell
python utils/prepare/preprocess.py  -data_path Dataset/Chest
```
which will automatically result in annotation json files in *./data/chest*

## Demo
We provide prediction demos of our models. The demo images can be picked from test dataset, you can simply run demo.py by using pretrained models:
```shell
python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset chest --load_from PRETRAINED_MODEL.pth --img_dir utils/demo_images
```
which will output like this:
```shell
utils/demo_images/000001.jpg prediction: Pneumonia, Atelectasis,
utils/demo_images/000004.jpg prediction: Fibrosis,
...
```

## Validation
We provide pretrained models on [Google Drive](https://www.google.com/drive/) for validation. ResNet101 trained on ImageNet with **CutMix** augmentation can be downloaded 
[here](https://drive.google.com/file/d/1seXPipXSgH_Fzk18vaVmclrXOIeocmfZ/view?usp=sharing).

For chest, run the following validation example:
```shell
python val.py --num_heads 1 --lam 0.1 --dataset chest --num_cls 14  --load_from MODEL.pth
```

## Training
#### Chest
You can run either of these two lines below 
```shell
python main.py --num_heads 1 --lam 0.1 --dataset chest --num_cls 14
python main.py --num_heads 1 --lam 0.1 --dataset chest --num_cls 14 --cutmix CutMix_ResNet101.pth
```
Note that the first command uses the Official ResNet-101 backbone while the second command uses the ResNet-101 pretrained on ImageNet with CutMix augmentation.