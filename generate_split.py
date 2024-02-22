import os
import json
import random


def generate_split(split: list[int], parent_folder='motion_data'):
    
    trainset, testset, valset = [], [], []

    for sess in os.listdir(parent_folder):
        sess_folder = os.path.join(parent_folder, sess)
        sess_idx = sess[0]
        for file in os.listdir(sess_folder):
            if file.endswith('.json'):
                example_idx = file[0]

                category = random.choices(['train', 'val', 'test'], split)[0]
                if category == 'train':
                    trainset.append([sess_idx, example_idx])
                elif category == 'val':
                    valset.append([sess_idx, example_idx])
                else:
                    testset.append([sess_idx, example_idx])
                    
    random.shuffle(trainset)
    random.shuffle(valset)
    random.shuffle(testset)

    with open('dataset_split.json', 'w') as f:
        dataset_split = { "train": trainset, "val": valset, "test": testset }
        json.load(dataset_split, f)
    
                

if __name__ == '__name__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        generate_split(config['train_val_test_split'])
