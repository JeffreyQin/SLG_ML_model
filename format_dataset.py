import os
import re
import json
import numpy as np
from tqdm import tqdm


def preproc_label(raw_label_file):
    with open(raw_label_file, 'r') as f:
        label = f.read()
        label = label.lower()
        label = re.sub(r'[^a-zA-Z0-9\s]', '', label)
        return label


def format_dataset(raw_folder='raw_data', formatted_folder='motion_data'):

    if not os.path.isdir(formatted_folder):
        os.mkdir(formatted_folder)

    for sess_idx, sess in tqdm(enumerate(os.listdir(raw_folder))):
        sess_folder = os.path.join(raw_folder, sess)
        formatted_sess_folder = os.path.join(formatted_folder, 'session_' + str(sess_idx))
        if not os.path.isdir(formatted_sess_folder):
            os.mkdir(formatted_sess_folder)
        for example_idx, example in enumerate(os.listdir(sess_folder)):
            input_file = os.path.join(sess_folder, example, 'data.csv')
            label_file = os.path.join(sess_folder, example, 'text.txt')

            input_data = np.genfromtxt(input_file, delimiter=',', usecols=tuple(range(1,9)))
            np.save(os.path.join(formatted_sess_folder, str(example_idx) + '.npy'), input_data)

            processed_label = preproc_label(label_file)
            with open(os.path.join(formatted_sess_folder, str(example_idx) + '.txt'), 'w') as f:
                f.write(processed_label)
            
            with open(os.path.join(formatted_sess_folder, str(example_idx) + '.json'), 'w') as f:
                data = {
                    "session": sess_idx,
                    "label": processed_label
                }
                json.dump(data, f)



if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        format_dataset(config['raw_data_folder'])