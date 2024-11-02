# Domain Adaptation for Crack Segmentation
Unsupervised Domain Adaptation method using Bayesian Neural Network for crack segmentation. [^1]


![全体像](https://github.com/toshimickey/domain-adaptation-for-crack-segmentation/assets/84856948/8e7b55ec-2d5c-4655-9734-ac31883fd754)



## How to prepare dataset
1. Download Crack Dataset from below [^2]:  
https://data.lib.vt.edu/articles/dataset/Concrete_Crack_Conglomerate_Dataset/16625056/1

2. Then, download our Dataset from below [^3]:  
https://sites.google.com/view/pchun/

3. Place the downloaded folders inside the `data/` directory, then run `DataPreprocessor.py` to set up the dataset.
    ```
    python　DataPreprocessor.py
    ```


## How to use virtual environment　with Miniconda
1. Install Miniconda
   - Download and install Miniconda from the official site: https://docs.conda.io/en/latest/miniconda.html  
   - Follow the installation instructions for your operating system.
2. Create a new environment:
    ```bash
    conda create -n YOUR-ENV-NAME python=3.11
    ```
3. Activate the environment:
    ```bash
    conda activate YOUR-ENV-NAME
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

[^1]:Chun, P.-J., & Kikuta, T. (2024). Self-training with Bayesian neural networks and spatial priors for unsupervised domain adaptation in crack segmentation. Computer-Aided Civil and Infrastructure Engineering, 39, 2642–2661. https://doi.org/10.1111/mice.13315
[^2]:Bianchi, Eric; Hebdon, Matthew (2021). Concrete Crack Conglomerate Dataset. University Libraries, Virginia Tech. Dataset. https://doi.org/10.7294/16625056.v1
[^3]:Pang-jo Chun, data sharing page https://sites.google.com/view/pchun/