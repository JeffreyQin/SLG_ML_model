import torch
from torch import utils, nn
import numpy as np

from preprocess import Tokenizer, preproc_timeseries

import os, sys, json



class MotionDataset(utils.data.Dataset):
    def __init__(self, test=False, val=False, token_type='char', data_folder='motion_data'):
        
        self.tokenizer = Tokenizer()

        with open('dataset_split.json', 'r') as dataset_split:
            data = json.load(dataset_split)
            if not test and not val:
                self.indices = data['train']
            elif test:
                self.indices = data['test']
            else:
                self.indices = data['val']

        inputs, labels = [], []
        for idx in self.indices:
            input = np.load(os.path.join(data_folder, str(idx) + '.npy'))
            with open(os.path.join(data_folder, str(idx) + '.txt'), 'r') as label_file:
                label = label_file.read()
            inputs.append(input)
            labels.append(label)

        self.X = preproc_timeseries(inputs)
        if token_type == 'char':
            embeddings, lengths, vocabs = self.tokenizer.char_tokenize(labels)
        else:
            embeddings, lengths, vocabs = self.tokenizer.subword_tokenize(labels)
        self.Y = nn.utils.rnn.pad_sequence([torch.tensor(embedding) for embedding in embeddings], batch_first=True)
        # self.Y = nn.utils.rnn.pack_padded_sequence(self.Y, lengths, batch_first=True, enforce_sorted=False)
        self.vocabs = vocabs


        self.X, self.Y = torch.tensor(np.array(self.X)), torch.tensor(np.array(self.Y))
        self.X, self.Y = self.X.type(torch.float32), self.Y.type(torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__():
        pass

