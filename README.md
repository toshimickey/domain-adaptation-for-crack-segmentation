# Domain Adaptation for Crack Segmentation
Unsupervised Domain Adaptation method using Bayesian Neural Network for crack segmentation.

## How to prepare dataset
Download Crack Dataset from below:
https://data.lib.vt.edu/articles/dataset/Concrete_Crack_Conglomerate_Dataset/16625056/1

Then, transform dataset 

```
unzip data.zip
python3 transform.py
```

## How to use virtual environment

```
python3 -m venv env
source env/bin/activate
pip install -r requiremments.txt
deactivate
```
