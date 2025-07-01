# MSRNet
MSRNet: A Multi-Scale Recursive Network for Camouflaged Object Detection



## MSRNet Performance Results

| Backbone        | CAMO-TE |                      |       |           |       |CHAMELEON |                      |       | COD10K-TE |                      |       | NC4K  |                      |       |
| --------------- | ------- | -------------------- | ----- | ----------|-------|--------- | -------------------- | ----- | --------- | -------------------- | ----- | ----- | -------------------- | ----- |
|                 | $S_m$   | $F^{\omega}_{\beta}$ | MAE   |$F_{\beta}$|$E_{m}$|$S_m$     | $F^{\omega}_{\beta}$ | MAE   | $S_m$     | $F^{\omega}_{\beta}$ | MAE   | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |
| ResNet-50       | 0.816   | 0.754                | 0.071 |           |       |0.908     | 0.858                | 0.021 | 0.861     | 0.768                | 0.026 | 0.874 | 0.816                | 0.037 |
| EfficientNet-B1 | 0.848   | 0.803                | 0.056 |           |       |0.916     | 0.870                | 0.020 | 0.863     | 0.773                | 0.024 | 0.876 | 0.823                | 0.036 |
| EfficientNet-B4 | 0.867   | 0.824                | 0.046 |           |       |0.911     | 0.865                | 0.020 | 0.875     | 0.797                | 0.021 | 0.884 | 0.837                | 0.032 |
| PVTv2-B2        | 0.874   | 0.839                | 0.047 |           |       |0.922     | 0.884                | 0.017 | 0.887     | 0.818                | 0.019 | 0.892 | 0.852                | 0.030 |
| PVTv2-B3        | 0.885   | 0.854                | 0.042 |           |       |0.927     | 0.898                | 0.017 | 0.895     | 0.829                | 0.018 | 0.900 | 0.861                | 0.028 |
| PVTv2-B4        | 0.888   | 0.859                | 0.040 |           |       |0.925     | 0.897                | 0.016 | 0.898     | 0.838                | 0.017 | 0.900 | 0.865                | 0.028 |
| PVTv2-B5        | 0.889   | 0.857                | 0.041 |           |       |0.924     | 0.885                | 0.018 | 0.898     | 0.827                | 0.018 | 0.903 | 0.863                | 0.028 |


## Preparing Datasets
In this research, we utilized four benchmark datasets for camouflaged object detection (CAMO, CHAMELEON, COD10K, NC4K).

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

