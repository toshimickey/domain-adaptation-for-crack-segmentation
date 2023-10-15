# Domain Adaptation for Crack Segmentation
Unsupervised Domain Adaptation method using Bayesian Neural Network for crack segmentation.

## How to prepare dataset
1. Download Crack Dataset from below [^1]:  
https://data.lib.vt.edu/articles/dataset/Concrete_Crack_Conglomerate_Dataset/16625056/1

2. Then, download our Dataset from below [^2]:  
https://drive.google.com/drive/folders/1pTUGTx88oTlIm3lsbdo3DKRqPadLrxD0?usp=sharing  
https://drive.google.com/drive/folders/1sRKRFanXP4gbA0WkoEMzT7OWCSRonxPG?usp=sharing

3. Set up the dataset using `dataloader/dataset.py` as a reference. You can also specify the original dataset.


## How to use virtual environment

```
python3 -m venv env
source env/bin/activate
pip install -r requiremments.txt
deactivate
```

[^1]:Bianchi, Eric; Hebdon, Matthew (2021). Concrete Crack Conglomerate Dataset. University Libraries, Virginia Tech. Dataset. https://doi.org/10.7294/16625056.v1
[^2]:Pang-jo Chun, data sharing page https://sites.google.com/view/pchun/
