import os
import json
import random


def generate_split(split: list[int], folder='motion_data'):
    
    trainset, testset, valset = [], [], []

    for file in os.listdir(folder):
        if file.endswith('.json'):
            data_idx = int(file[0])

            category = random.choices(['train', 'val', 'test'], split)[0]
            if category == 'train':
                trainset.append(data_idx)
            elif category == 'val':
                valset.append(data_idx)
            else:
                testset.append(data_idx)
    
    random.shuffle(trainset)
    random.shuffle(valset)
    random.shuffle(testset)

    with open('dataset_split.json', 'w') as f:
        dataset_split = { "train": trainset, "val": valset, "test": testset }
        json.dump(dataset_split, f)
    
                

if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        generate_split(config['train_val_test_split'])
