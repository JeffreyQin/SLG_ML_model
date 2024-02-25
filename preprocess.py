import torchtext
import torch
import numpy as np

# expected shape: [# timestamp, # channel]
def z_score_normalize(input):
    channel_mean = np.mean(input, axis=0)
    channel_std = np.std(input, axis=0)
    input = (input / channel_mean) / channel_std
    return input

# expected shape: [# timestamp, # channel]
def moving_average(input, window_size=5):
    kernel = np.ones(window_size) / window_size
    convolved = np.zeros(shape=(input.shape[0] - window_size + 1, input.shape[1]))
    for channel in range(input.shape[1]):
        convolved[:,channel] = np.convolve(input[:,channel], kernel, mode='valid')
    return convolved

def unity_length(input, length=100):
    pass

def crop_invalid(input):
    return input[1:]

# expected shape: [batch, # timestamp, # channel]
def preproc_timeseries(inputs):

    for idx, input in enumerate(inputs):
        input = crop_invalid(input)
        input = z_score_normalize(input)
        input = moving_average(input, window_size=5)
        # input = unify_length(input, length=100)
        inputs[idx] = input

    return inputs


class Tokenizer(object):
    def __init__(self):
        pass

    def char_tokenize(self, labels):
        tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=(lambda text: list(text)), language='en')

        def yield_tokens(texts):
            for text in texts:
                yield tokenizer(text)
        self.vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(labels), min_freq=1, specials=['<pad>'])
        
        embeddings = [[self.vocab[token] for token in tokenizer(label)] for label in labels]
        lengths = [len(embedding) for embedding in embeddings]
        return embeddings, lengths, self.vocab

    def subword_tokenizer():
        pass