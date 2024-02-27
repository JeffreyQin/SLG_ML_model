import torchtext
import torch
import numpy as np
import os, json

# expected shape: [# timestamp, # feature]
def z_score_normalize(input):
    mean_per_feature = np.mean(input, axis=0)
    std_per_feature = np.std(input, axis=0)
    input = (input / mean_per_feature) / std_per_feature
    return input

# expected shape: [# timestamp, # feature]
def moving_average(input, window_size=5):
    kernel = np.ones(window_size) / window_size
    convolved = np.zeros(shape=(input.shape[0] - window_size + 1, input.shape[1]))

    for feature in range(input.shape[1]):
        convolved[:,feature] = np.convolve(input[:,feature], kernel, mode='valid')

    return convolved

def unity_length(input, length=100):
    pass

def crop_invalid(input):
    return input[1:]

# expected shape: [batch, # timestamp, # feature]
def preproc_timeseries(inputs):

    for idx, input in enumerate(inputs):
        input = crop_invalid(input)
        input = z_score_normalize(input)
        input = moving_average(input, window_size=5)
        # input = unify_length(input, length=100)
        inputs[idx] = input

    return inputs


class Tokenizer(object):
    def __init__(self, type='char'):

        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            data_folder = config['formatted_data_folder']
            num_examples = len(os.listdir(data_folder)) // 3

        self.type = type

        if self.type == 'char':
            labels = []
            for idx in range(num_examples):
                with open(os.path.join(data_folder, str(idx) + '.txt'), 'r') as label_file:
                    label = label_file.read()
                    labels.append(label)

            tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=lambda text: list(text), language='en')

            def yield_tokens(texts):
                for text in texts:
                    yield tokenizer(text)
            
            self.vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(labels), min_freq=1, specials=['<pad>'])
            self.embeddings = [[self.vocab[token] for token in tokenizer(label)] for label in labels]
            self.lengths = [len(embedding) for embedding in self.embeddings]
        else:
            pass

    def get_tokenized(self, indices):
        embeddings = [self.embeddings[idx] for idx in indices]
        lengths = [self.lengths[idx] for idx in indices]
        return embeddings, lengths, self.vocab
    
    def decode_embeddings(self, embedding):
        pass