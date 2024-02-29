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

        labels = []
        for idx in range(num_examples):
            with open(os.path.join(data_folder, str(idx) + '.txt'), 'r') as label_file:
                label = label_file.read()
                labels.append(label)

        self.type = type
        
        if self.type == 'subword':
            pass

        else:
            def yield_tokens(texts):
                for text in texts:
                    yield self.tokenizer(text)

            if self.type == 'char':
                self.tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=lambda text: list(text), language='en')
            elif self.type == 'word':
                self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
            
            self.vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(labels), min_freq=1, specials=['<pad>'])
            self.text_to_int = self.vocab.get_stoi()
            self.int_to_text = {int(value): str(key) for key, value in self.text_to_int.items()}
            

    def create_vocab_file(self, path='vocab.json'):
        with open(path, 'w') as f:
            json.dump({
                "text_to_int": self.text_to_int,
                "int_to_text": self.int_to_text
            }, f)

    def get_tokenized(self, labels, vocab_file=None):
        
        # check if custom vocab is used
        if vocab_file is None:
            embeddings = [[self.text_to_int[token] for token in self.tokenizer(label)] for label in labels]
        else:
            with open(vocab_file, 'r') as f:
                text_to_int = json.load(vocab_file)['text_to_int']
            embeddings = [[text_to_int[token] for token in self.tokenizer(label)] for label in labels]
            
        lengths = [len(embedding) for embedding in embeddings]
        return embeddings, lengths
    
    
    def decode_tokenized(self, embeddings, vocab_file='vocab.json'):

        # check if custom vocab is used
        if vocab_file is None:
            texts = [[self.int_to_text[idx] for idx in embedding] for embedding in embeddings]
        else: 
            with open(vocab_file, 'r') as f:
                int_to_text = json.load(vocab_file)['int_to_text']
            texts = [[int_to_text[idx] for idx in embedding] for embedding in embeddings]
        
        return texts
