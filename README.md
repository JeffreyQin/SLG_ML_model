# Signal_Glove_Model

Official ML repo. for the sign language interpreter. 


## How to use

**Data preprocessing**

1. Put raw data in a folder called "raw_data", or edit your desired folder path in "config.json"
2. Run ```python format_dataset.py``` to re-format your data into a folder called "motion_data"
3. Run ```python generate_split.py``` to generate train/validation/testing split

**Running model**

1. To train a model, run ```python train_model.py``` with the ```evaluate_saved``` flag set to None.
2. To evaluate a trained model, run ```python train_model.py``` with the ```evaluate_saved``` flag appointed to your desired model path.
