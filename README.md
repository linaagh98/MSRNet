# MSRNet
MSRNet: A Multi-Scale Recursive Network for Camouflaged Object Detection

![Methodology](https://github.com/linaagh98/MSRNet/blob/main/images/Methodology%20Diagram.png?raw=true)

## MSRNet Performance Results

| Backbone        | CAMO  |                      |       |           |       |CHAMELEON |                      |       |           |       | COD10K |                      |       |           |       | NC4K  |                      |       |           |       |
| --------------- | ----- | -------------------- | ----- | ----------|-------|--------- | -------------------- | ----- | ----------|-------| -------| -------------------- | ----- | ----------|-------| ----- | -------------------- | ----- |---------- |-------|
|                 | $S_m$ | $F^{\omega}_{\beta}$ | MAE   |$F_{\beta}$|$E_{m}$|$S_m$     | $F^{\omega}_{\beta}$ | MAE   |$F_{\beta}$|$E_{m}$| $S_m$  | $F^{\omega}_{\beta}$ | MAE   |$F_{\beta}$|$E_{m}$| $S_m$ | $F^{\omega}_{\beta}$ | MAE   |$F_{\beta}$|$E_{m}$|
| ResNet-50       | 0.816 | 0.754                | 0.071 | 0.794     | 0.872 |0.918     | 0.876                | 0.020 | 0.888     | 0.975 | 0.868  | 0.786                | 0.024 | 0.816     | 0.934 | 0.869 | 0.814                | 0.039 | 0.844     | 0.925 |
| EfficientNet-B4 | 0.875 | 0.838                | 0.045 | 0.863     | 0.936 |0.923     | 0.881                | 0.019 | 0.891     | 0.970 | 0.887  | 0.814                | 0.020 | 0.838     | 0.947 | 0.889 | 0.844                | 0.031 | 0.866     | 0.943 |
| PVTv2-B2        | 0.873 | 0.838                | 0.047 | 0.860     | 0.928 |0.931     | 0.904                | 0.016 | 0.912     | 0.976 | 0.894  | 0.829                | 0.018 | 0.849     | 0.952 | 0.894 | 0.853                | 0.030 | 0.874     | 0.943 |
| PVTv2-B3        | 0.885 | 0.855                | 0.043 | 0.874     | 0.941 |0.933     | 0.907                | 0.016 | 0.915     | 0.973 | 0.904  | 0.847                | 0.017 | 0.865     | 0.959 | 0.903 | 0.867                | 0.027 | 0.886     | 0.952 |
| PVTv2-B4        | 0.888 | 0.861                | 0.040 | 0.878     | 0.942 |0.932     | 0.908                | 0.017 | 0.916     | 0.978 | 0.907  | 0.852                | 0.016 | 0.868     | 0.962 | 0.905 | 0.873                | 0.026 | 0.890     | 0.953 |
| PVTv2-B5        | 0.888 | 0.860                | 0.041 | 0.876     | 0.943 |0.925     | 0.893                | 0.017 | 0.903     | 0.971 | 0.902  | 0.844                | 0.017 | 0.862     | 0.957 | 0.903 | 0.871                | 0.027 | 0.889     | 0.952 |


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

