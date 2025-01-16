# Degree FYP1
# CRAFT: Character-Region Awareness For Text detection
## Files
```bash
├── basenet
│   ├── __init__.py
│   ├── vgg16_bn.py
├── config
│   ├── load_config.py
│   ├── syn_train.yaml
│   ├── ic15_train.yaml
├── data
│   ├── SynthText
│   │   ├── gt.mat
│   ├── ICDAR2013
│   │   ├── Challenge2_Training_Task12_Images
│   │   └── Challenge2_Training_Task1_GT
│   │   ├── Challenge2_Test_Task12_Images
│   │   └── Challenge2_Test_Task1_GT
│   ├── ICDAR2015
│   │   ├── ch4_training_images
│   │   └── ch4_training_localization_transcription_gt
│   │   ├── ch4_test_images
│   │   └── ch4_test_localization_transcription_gt
│   ├── pseudo_label
│   │   ├── make_charbox.py
│   │   └── watershed.py
│   ├── boxEnlarge.py
│   ├── dataset.py
│   ├── gaussian.py
│   ├── imgaug.py
│   ├── imgproc.py
├── exp (This folder will generated once run the `trainSynth.py` or `trainIC15.py`)
│   ├── syn_train
│   │   ├── syn_train.yaml
│   │   └── syn_train-ic13-iou (This folder will generated once run the `ic13Eval.py`)
│   ├── ic15_train
│   │   ├── ic15_train.yaml
│   │   └── ic15_train-ic15-iou (This folder will generated once run the `ic15Eval.py`)
├── loss
│   └── mseloss.py
├── metrics
│   └── eval_det_iou.py
├── weights
│   ├── CRAFT_clr_amp_14000.pth
│   ├── CRAFT_clr_amp_29500.pth
│   ├── craft_ic15_20k.pth
│   ├── craft_ic15_25k.pth
│   └── craft_refiner_CTW1500.pth
├── images
├── result
├── craft.py
├── craft_utils.py
├── file_utils.py
├── util.py
├── inference_boxes.py
├── refinenet.py
├── trainSynth.py
├── trainIC15.py
├── ic13Eval.py
├── ic15Eval.py
├── eval.py (Shared by `ic13Eval.py` and `ic15Eval.py`)
├── test.py 
└── craft.ipynb
```
There are some files might need to modify the path to run the code (You may use `ctrl + F` to search "drive" as keyword to modify):
* `config/syn_train.yaml`
* `config/ic15_train.yaml`
* `trainSynth.py`
* `trainIC15.py`
* `eval.py`
* `test.py`
* `craft.ipynb`

## Requirements
Use T4 runtime in Google Colab, because it requires GPU to run the code.
### Libraries to install
```bash
black==19.10b0
conda==4.10.3
opencv-python==4.5.3.56
Pillow==8.2.0
PyYAML==5.4.1
scikit-image==0.17.2
Shapely==1.8.0
torch==1.9.0
torchvision==0.10.0
wandb==0.12.9
scipy==1.1.0
```

## Training
### Training on SynthText Dataset
```python
!python '/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/trainSynth.py'
```
### Training on SynthText Dataset
```python
!python '/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/trainIC15.py'
```

### Evaluation of ICDAR 2013

## Evaluation
### Evaluation of ICDAR 2013
```python
!python '/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/ic13Eval.py'
```
To evaluate using different models, go to `config/syn_train.yaml`, find `trained_model`, and change the path following the model name.

E.g. `trained_model: "/content/drive/MyDrive/FYP1/CRAFT/weights/craft_mlt_25k.pth"`
| Training Datasets                  | Model                   | Precision  | Recall     | H-mean     |
| ---------------------------------- | ----------------------- | :--------: | :--------: | :--------: |
| SynthText                          | CRAFT_clr_amp_29500.pth | 0.7642     | 0.7516     | 0.7578     |
| SynthText + ICDAR2015              | CRAFT_clr_amp_14000.pth | 0.8548     | 0.8009     | 0.8270     |
| SynthText + ICDAR2013 + ICDAR 2017 | craft_mlt_25k.pth       | 0.9025     | 0.8877     | 0.8950     |

### Evaluation of ICDAR 2015
```python
!python '/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/ic15Eval.py'
```
To evaluate using different models, go to `config/ic15_train.yaml`, find `trained_model`, and change the path following the model name.

E.g. `trained_model: "/content/drive/MyDrive/FYP1/CRAFT/weights/craft_mlt_25k.pth"`
| Training Datasets                  | Model                   | Precision  | Recall     | H-mean     |
| ---------------------------------- | ----------------------- | :--------: | :--------: | :--------: |
| SynthText                          | CRAFT_clr_amp_29500.pth | 0.6231     | 0.6066     | 0.6148     |
| SynthText + ICDAR2015              | CRAFT_clr_amp_14000.pth | 0.8811     | 0.8238     | 0.8515     |
| SynthText + ICDAR2013 + ICDAR 2017 | craft_mlt_25k.pth       | 0.8388     | 0.8315     | 0.8351     |

## Test instruction
* `--trained_model` : weight file.
* `--test_folder` : folder path to test images.
```python
!python '/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/test.py' --trained_model='/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/weights/craft_mlt_25k.pth' --test_folder='/content/drive/My Drive/Lim Wee Zheng/Source Code/Main Folder/images'
```
### Generated Result
3 files will be generated in the `result` folder.
* Heatmap mask
* Generated detection inference boxes
* Location of inference boxes
