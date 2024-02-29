
import torch
from torch import utils, nn
import numpy as np

from preprocess import Tokenizer, preproc_timeseries

import os, sys, json


class MotionDataset(utils.data.Dataset):
    def __init__(self, tokenizer, test=False, val=False):
        
        self.tokenizer = tokenizer
        
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            data_folder = config['formatted_data_folder']

        with open('dataset_split.json', 'r') as dataset_split:
            data = json.load(dataset_split)
            if not test and not val:
                self.indices = data['train']
            elif test:
                self.indices = data['test']
            else:
                self.indices = data['val']

        if len(self.indices) == 0:
            return
        
        inputs, labels = [], []
        for idx in self.indices:
            input = np.load(os.path.join(data_folder, str(idx) + '.npy'))
            with open(os.path.join(data_folder, str(idx) + '.txt'), 'r') as f:
                label = f.read()
            inputs.append(input)
            labels.append(label)

        self.X = preproc_timeseries(inputs)

        self.embeddings, self.lengths = self.tokenizer.get_tokenized(labels)
        self.Y = nn.utils.rnn.pad_sequence([torch.tensor(embedding) for embedding in self.embeddings], batch_first=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        X = torch.tensor(self.X[index]).type(torch.float32)
        Y = self.Y[index].type(torch.float32)
        len = torch.tensor(self.lengths[index]).type(torch.int64)
        return X, Y, len

