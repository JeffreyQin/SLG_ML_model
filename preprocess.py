import torchtext
import torch
import numpy as np

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