# Signal_Glove_Model

Official ML repo. for the sign language interpreter. 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


## How to use

**Data preprocessing**

1. Put raw data in a folder called "raw_data", or edit your desired folder path in "config.json"
2. Run ```python format_dataset.py``` to re-format your data into a folder called "motion_data"
3. Run ```python generate_split.py``` to generate train/validation/testing split

**Running model**

1. To train a model, run ```python train_model.py``` with the ```evaluate_saved``` flag set to None.
2. To evaluate a trained model, run ```python train_model.py``` with the ```evaluate_saved``` flag appointed to your desired model path.
