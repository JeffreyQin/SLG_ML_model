import torch
from torch import nn, utils, optim
import numpy as np

from dataset import MotionDataset
from architecture import LSTMModel
from preprocess import Tokenizer

from tqdm import tqdm
from absl import flags
import os, sys
import logging


FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('model_folder', 'models', '')
flags.DEFINE_string('evaluate_saved', None, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('train_epochs', 10, '')
flags.DEFINE_integer('learning_rate', 1e-3, '')
flags.DEFINE_string('tokenizer', 'char', '')

def evaluate(eval_dataset, model, device):

    dataloader = utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
    vocab_size = len(eval_dataset.tokenizer.vocab)
    

    model.to(device)
    model.eval()
    with torch.no_grad():
        for example_idx, (exampleX, exampleY, exampleY_len) in tqdm(enumerate(dataloader)):

            exampleX, exampleY = exampleX.to(device), exampleY.to(device)
            
            pred = model(exampleX)
    


def train(train_dataset, val_dataset, device):

    dataloader = utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    vocab_size = len(train_dataset.tokenizer.vocab)

    model = LSTMModel(input_size=8, output_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    ctc_loss_fn = nn.CTCLoss(blank=0) # 0 (blank) is the vocab index of <pad>

    best_epoch_loss = float('inf')
    best_epoch_acc = float('-inf')

    for epoch_idx in range(FLAGS.train_epochs):
        model.train()
        batch_losses = []
        for batch_idx, (batchX, batchY, batchY_len) in tqdm(enumerate(dataloader)):
            batchX, batchY = batchX.to(device), batchY.to(device)

            pred = model(batchX)
            pred_len = torch.full(size=FLAGS.batch_size, fill_value=pred.size(0), dtype=torch.long)
            loss = ctc_loss_fn(pred, batchY, pred_len, batchY_len)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.detach().numpy())

        epoch_loss = np.mean(batch_losses)
        epoch_acc = evaluate(val_dataset, model, device)
        logging.info(f'completed epoch: {epoch_idx + 1}, average loss: {epoch_loss}, accuracy: {epoch_acc}')

        if epoch_loss < best_epoch_loss:
            torch.save(model.state_dict, os.path.join(FLAGS.model_folder, 'best_loss_model.pt'))
        if epoch_acc > best_epoch_acc:
            torch.save(model.state_dict, os.path.join(FLAGS.model_folder, 'best_acc_model.pt'))

        torch.save(model.state_dict, os.path.join(FLAGS.model_folder, 'model.pt'))

    return model



def main():

    logging.basicConfig(level=logging.INFO, format='%{message}s')
    logging.info('process started')

    if torch.cuda.is_available():
        logging.info('cuda device available')
        device = 'cuda'
    else:
        logging.info('cuda device NOT available')
        device = 'cpu'


    if FLAGS.tokenizer == 'char':
        tokenizer = Tokenizer(type='char')
    else:
        tokenizer = Tokenizer(type='subword')

    if FLAGS.evaluate_saved is not None:
        testset = MotionDataset(tokenizer, test=True)
        model = torch.load(FLAGS.evaluate_saved)
        accuracy = evaluate(testset, model, device)
        logging.info('evaluation complete')
    else:
        trainset = MotionDataset()
        valset = MotionDataset(tokenizer, val=True)
        model = train(trainset, valset, device)
        logging.info('training complete')
    

if __name__ == '__main__':
    FLAGS(sys.args)
    main()