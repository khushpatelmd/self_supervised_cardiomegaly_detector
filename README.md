# Interpretable Self-supervised Learning for Detection of Cardiomegaly from Chest X-Ray Images 

### Background: 
Pattern recognition of different diseases from medical imaging studies using deep learning is evolving rapidly, with some algorithms performing better than expert radiologists in identifying these diseases. One area where deep learning algorithms could improve clinical workflows in a variety of medical settings is in automated cardiomegaly detection from chest X-ray images. Biomedical datasets are highly imbalanced with only few images for diseased patients and wide number of images for healthy subjects. To overcome this limitation, we developed three self-supervised approaches. A wide number of Model Interpretability approaches were used to develop trust in the deep learning model. 

<hr />

# Table Of Contents
-  [Training strategies](#Training-strategies)
-  [Interpretation](#Interpretation)
-  [How to run the code](#How-to-run)
-  [Code structure](#Code-structure)
-  [Manuscript](#Manuscript)
-  [Requirements](#Requirements)
-  [How to cite](#How-to-cite)

<hr />

# Training strategies

Detailed description of training can be seen in the manuscript. Self supervised training approaches used included proxy tasks of age and sex and image reconstruction tasks. 

Self supervised approach

<img src="files/Proxy.png" width=800 align="center">

Unsupervised approach

<img src="files/proxy2.png" width=800 align="center">

Learning rate optimization

<img src="files/learning_rate_optimization.png" width=800 align="center">


# Interpretation

Interpretation of deep learning models generally considered as blackbox is vital for biomedical problems. 

<img src="files/GRAD-CAM.png" width=800 align="center">

<img src="files/GRAD-CAM2.png" width=800 align="center">

# How to run

# <hr />

# Code structure
```
├──  configs
│    └── config.py - change data/logging/checkpoint paths, experiment name, gpu, training options, hyperparameters
│
├──  data  
│    └── dataset_MLM_NSP.py - dataset class for MLM and NSP pretraining tasks
│    └── dataset_MLM.py - dataset class for MLM pretraining tasks
│
├──  engine - The training function to be used in the training files.
│   ├── engine_amp_MLM_NSP.py  - Main training loop for MLM and NSP task to be used inside train_*.py files
│   └── engine_amp_MLM.py  - Main training loop for MLM task
│
├── train - this folder contains main training files. 
│   └── train_dynamic_mask.py - Main training file for dynamic MLM strategy
│   └── train_fixed_mask_nsp.py - Main training file for fixed MLM strategy and NSP task
│   └── train_fixed_mask.py - Main training file for fixed MLM strategy 
│   └── resume_train_dynamic_mask.py - resuming training file for dynamic MLM strategy from last checkpoint
│   └── resume_train_fixed_mask_nsp.py - resuming training file for fixed MLM strategy and NSP task from last checkpoint
│   └── resume_train_fixed_mask.py - resuming training file for fixed MLM strategy from last checkpoint
│ 
└── utils
     └── utils.py - misc utils 
     └── requirements.txt - python libraries
     
```
<hr />

# Requirements
The `requirements.txt` file contains all Python libraries and they will be installed using:
```
pip install -r requirements.txt
```
absl-py==1.0.0
aiohttp==3.8.1
aiosignal==1.2.0
albumentations==1.1.0
async-timeout==4.0.1
attrs==21.2.0
cachetools==4.2.4
certifi==2021.10.8
charset-normalizer==2.0.7
cycler==0.11.0
fonttools==4.28.2
frozenlist==1.2.0
fsspec==2021.11.0
future==0.18.2
google-auth==2.3.3
google-auth-oauthlib==0.4.6
grpcio==1.42.0
idna==3.3
imageio==2.11.1
importlib-metadata==4.8.2
joblib==1.1.0
kiwisolver==1.3.2
Markdown==3.3.6
matplotlib==3.5.0
mkl-fft==1.3.1
mkl-random @ file:///tmp/build/80754af9/mkl_random_1626186066731/work
mkl-service==2.4.0
monai==0.7.0
multidict==5.2.0
networkx==2.6.3
numpy @ file:///tmp/build/80754af9/numpy_and_numpy_base_1634095651905/work
oauthlib==3.1.1
olefile @ file:///Users/ktietz/demo/mc3/conda-bld/olefile_1629805411829/work
opencv-python-headless==4.5.4.60
packaging==21.3
pandas==1.3.4
Pillow==8.4.0
protobuf==3.19.1
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyDeprecate==0.3.1
pyparsing==3.0.6
python-dateutil==2.8.2
pytorch-lightning==1.5.0
pytz==2021.3
PyWavelets==1.2.0
PyYAML==6.0
qudida==0.0.4
requests==2.26.0
requests-oauthlib==1.3.0
rsa==4.7.2
scikit-image==0.18.3
scikit-learn==1.0.1
scipy==1.7.2
setuptools-scm==6.3.2
six @ file:///tmp/build/80754af9/six_1623709665295/work
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
threadpoolctl==3.0.0
tifffile==2021.11.2
tomli==1.2.2
torch==1.10.0
torchaudio==0.10.0
torchmetrics==0.6.0
torchvision==0.11.1
tqdm==4.62.3
typing-extensions @ file:///tmp/build/80754af9/typing_extensions_1631814937681/work
urllib3==1.26.7
Werkzeug==2.0.2
yarl==1.7.2
zipp==3.6.0
~~~
<hr />

# How to cite
This repository is a research work in progress. Please contact author (drpatelkhush@gmail.com) for details on reuse of code.


