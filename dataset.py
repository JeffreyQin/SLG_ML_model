import torch
from torch import utils, nn
import numpy as np

from preprocess import Tokenizer, preproc_timeseries

import os, sys, json



class MotionDataset(utils.data.Dataset):
    def __init__(self, test=False, val=False, tokenizer):
        
        self.tokenizer = tokenizer

        with open('dataset_split.json', 'r') as dataset_split:
            data = json.load(dataset_split)
            if not test and not val:
                self.indices = data['train']
            elif test:
                self.indices = data['test']
            else:
                self.indices = data['val']

        inputs = []
        for idx in self.indices:
            input = np.load(os.path.join(data_folder, str(idx) + '.npy'))
            inputs.append(input)

        self.X = preproc_timeseries(inputs)
        embeddings, self.lengths, self.vocab = self.tokenizer.get_tokenized(self.indices)
        self.Y = nn.utils.rnn.pad_sequence([torch.tensor(embedding) for embedding in embeddings], batch_first=True)
        
        self.X, self.Y = torch.tensor(np.array(self.X)), torch.tensor(np.array(self.Y))
        self.X, self.Y = self.X.type(torch.float32), self.Y.type(torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.lengths[index]

