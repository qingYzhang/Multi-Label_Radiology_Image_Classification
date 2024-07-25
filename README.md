## Requirements
- Python
- pytorch
- torchvision
- pillow
- scikit-learn

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
python demo.py --model resnet101 --load_from PRETRAINED_MODEL.pth --img_dir utils/demo_images
```
which will output like this:
```shell
utils/demo_images/000001.jpg prediction: Pneumonia, Atelectasis,
utils/demo_images/000004.jpg prediction: Fibrosis,
...
```

## Validation
Run the following validation example:
```shell
python val.py --load_from MODEL.pth
```

## Training
Run the line below 
```shell
python main_final.py --model ${model}
```