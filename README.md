# MSRNet
MSRNet: A Multi-Scale Recursive Network for Camouflaged Object Detection

## Preparing Datasets
In this research, we utilized four benchmark datasets for camouflaged object detection (CAMO, CHAMELEON, COD10K, NC4K). 
You can download the dataset from the following links: 

CAMO: https://www.kaggle.com/datasets/ivanomelchenkoim11/camo-dataset

CHAMELEON:

COD10K: https://www.kaggle.com/datasets/getcam/cod10k?resource=download

NC4K:

After downloading all datasets, you need to create a file named "dataset.yaml" and place it in the same directory as the main code folder.  
The dataset.yaml file will include the paths for your Train and Test datasets. Please ensure that you place the datasets in the corresponding paths as you specified in the dataset.yaml file. 
Your dataset.yaml file should look something like this:

```yaml
# ICOD Datasets
cod10k_tr:
  {
    root: "YOUR_ROOT_DIRECTRY/ICOD_Datasets/Train/COD10K-TR",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
camo_tr:
  {
    root: "YOUR_ROOT_DIRECTRY/ICOD_Datasets/Train/CAMO-TR",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
cod10k_te:
  {
    root: "YOUR_ROOT_DIRECTRY/ICOD_Datasets/Test/COD10K-TE",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
camo_te:
  {
    root: "YOUR_ROOT_DIRECTRY/ICOD_Datasets/Test/CAMO-TE",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
chameleon:
  {
    root: "YOUR_ROOT_DIRECTRY/ICOD_Datasets/Test/CHAMELEON",
    image: { path: "Image", suffix: ".jpg" },
    mask: { path: "Mask", suffix: ".png" },
  }
nc4k:
  {
    root: "YOUR_ROOT_DIRECTRY/ICOD_Datasets/Test/NC4K",
    image: { path: "Imgs", suffix: ".jpg" },
    mask: { path: "GT", suffix: ".png" },
  }
```


## Install Requirements

* torch==2.1.2
* torchvision==0.16.2
* Others: `pip install -r requirements.txt`


## Evaluation

```shell
# ICOD
python main_for_image.py --config configs/icod_train.py --model-name <MODEL_NAME> --evaluate --load-from <TRAINED_WEIGHT>
# VCOD
python main_for_video.py --config configs/vcod_finetune.py --model-name <MODEL_NAME> --evaluate --load-from <TRAINED_WEIGHT>
```


## Training

### Image Camouflaged Object Detection

```shell
python main_for_image.py --config configs/icod_train.py --pretrained --model-name EffB1_MSRNet
python main_for_image.py --config configs/icod_train.py --pretrained --model-name EffB4_MSRNet
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B2_MSRNet
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B3_MSRNet
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B4_MSRNet
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B5_MSRNet
python main_for_image.py --config configs/icod_train.py --pretrained --model-name RN50_MSRNet
```
> [!note]
> These command-lines will not save the final predection images of the trained model, to save the predection sesults of your traind model add --save-results to your command-line. 

