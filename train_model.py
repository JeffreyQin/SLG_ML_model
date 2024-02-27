import torch
from torch import nn, utils, optim

from dataset import MotionDataset
from architecture import Model

from absl import flags
import os, sys
import logging


FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_string('evaluate_saved', None, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('train_epochs', 10, '')
flags.DEFINE_integer('learning_rate', 1e-3, '')
flags.DEFINE_string('tokenizer', 'char', '')

def evaluate():
    pass

def train():
    pass

def main():
    logging.basicConfig(level=logging.INFO, format='%{message}s')
    logging.info('process started')

    if torch.cuda.is_available():
        logging.info('cuda device available')
        device = 'cuda'
    else:
        logging.info('cuda device NOT available')
        device = 'cpu'

    
    
    if FLAGS.evaluate_saved is not None:
        testset = MotionDataset(test=True)
        evaluate(testset, device)
        logging.info('evaluation complete')
    else:
        trainset = MotionDataset()
        valset = MotionDataset(val=True)
        train(trainset, valset, device)
        logging.info('training complete')
    

if __name__ == '__main__':
    FLAGS(sys.args)
    main()