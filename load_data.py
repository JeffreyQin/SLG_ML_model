import os, sys
import re
import json
import random
import numpy as np
from tqdm import tqdm

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('raw_data_folder', 'raw_data', '')
flags.DEFINE_string('formatted_data_folder', 'motion_data', '')
flags.DEFINE_list('train_val_test_split', [0.8, 0.1, 0.1], '')


def generate_split(split: list[int], folder='motion_data'):
    
    trainset, testset, valset = [], [], []

    for file in os.listdir(folder):
        if file.endswith('.json'):
            data_idx = int(file.split('.')[0])

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
    
                
def preproc_label(raw_label_file):
    with open(raw_label_file, 'r') as f:
        label = f.read()
        label = label.lower()
        label = re.sub(r'[^a-zA-Z0-9\s]', '', label)
        return label


def format_dataset(raw_folder='raw_data', formatted_folder='motion_data'):

    if not os.path.isdir(formatted_folder):
        os.mkdir(formatted_folder)

    data_idx = 0

    for sess_idx, sess in tqdm(enumerate(os.listdir(raw_folder))):
        sess_folder = os.path.join(raw_folder, sess)
        for example_idx, example in enumerate(os.listdir(sess_folder)):
            input_file = os.path.join(sess_folder, example, 'data.csv')
            label_file = os.path.join(sess_folder, example, 'text.txt')

            input_data = np.genfromtxt(input_file, delimiter=',', usecols=tuple(range(1,9)))
            np.save(os.path.join(formatted_folder, str(example_idx) + '.npy'), input_data)

            processed_label = preproc_label(label_file)
            with open(os.path.join(formatted_folder, str(data_idx) + '.txt'), 'w') as f:
                f.write(processed_label)

            with open(os.path.join(formatted_folder, str(data_idx) + '.json'), 'w') as f:
                data = {
                    "index": data_idx,
                    "session": sess_idx,
                    "example": example_idx,
                    "label": processed_label
                }
                json.dump(data, f)
            
            data_idx += 1


if __name__ == '__main__':
    FLAGS(sys.argv)
    format_dataset(FLAGS.raw_data_folder, FLAGS.formatted_data_folder)
    generate_split(FLAGS.train_val_test_split, FLAGS.formatted_data_folder)