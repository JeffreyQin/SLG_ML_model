import torch
from torch import nn, utils, optim
import numpy as np
import jiwer

from dataset import MotionDataset
from architecture import LSTMModel
from preprocess import Tokenizer


from tqdm import tqdm
from absl import flags
import os, sys
import logging


FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_string('model_folder', 'models', '')
flags.DEFINE_string('vocab_file', 'vocab.json', '')
flags.DEFINE_string('evaluate_saved', None, '')
flags.DEFINE_integer('batch_size', 3, '')
flags.DEFINE_integer('train_epochs', 100, '')
flags.DEFINE_float('learning_rate', 1e-3, '')
flags.DEFINE_string('tokenizer', 'char', '')

def evaluate(eval_dataset, model, device):

    dataloader = utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
    vocab_size = eval_dataset.tokenizer.vocab_size
    
    model.to(device)
    model.eval()

    total_err = 0
    total_tokens = 0
    with torch.no_grad():
        for example_idx, (exampleX, exampleY, exampleY_len) in tqdm(enumerate(dataloader)):

            exampleX, exampleY = exampleX.to(device), exampleY.to(device)
            
            pred = model(exampleX).squeeze()
            pred = torch.argmax(pred, dim=1)

            exampleY = exampleY.squeeze()

            pred_text = eval_dataset.tokenizer.decode_tokenized(pred.cpu(), vocab_file=FLAGS.vocab_file)
            target_text = eval_dataset.tokenizer.decode_tokenized(exampleY, vocab_file=FLAGS.vocab_file)
            
            if FLAGS.debug:
                logging.info(pred_text)
                logging.info(target_text)

            if FLAGS.tokenizer == 'char':
                total_err += jiwer.cer(target_text, pred_text)
                total_tokens += len(target_text.split())
            else:
                total_err += jiwer.wer(target_text, pred_text)
                total_tokens += len(list(target_text))
    
    err_rate = total_err / total_tokens
    return err_rate
            

    
def train(train_dataset, val_dataset, device):

    dataloader = utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    vocab_size = train_dataset.tokenizer.vocab_size

    model = LSTMModel(input_size=8, output_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    CE_loss_fn = nn.CrossEntropyLoss()

    best_epoch_loss = float('inf')
    best_epoch_acc = float('-inf')

    for epoch_idx in range(FLAGS.train_epochs):
        model.train()
        batch_losses = []

        for batch_idx, (batchX, batchY, batchY_len) in tqdm(enumerate(dataloader)):
            batchX, batchY = batchX.to(device), batchY.to(device)

            pred = model(batchX) # shape [batch size, seq length, num features]
            pred_seq_len, Y_seq_len = pred.size(1), batchY.size(1)

            # apply padding
            if pred_seq_len > Y_seq_len:
                batchY = torch.cat([batchY, torch.zeros(size=(pred.size(0), pred_seq_len - Y_seq_len)).long()], dim=1)
            elif pred_seq_len < Y_seq_len:
                pred = torch.cat([pred, torch.zeros(size=(pred.size(0), Y_seq_len - pred_seq_len, pred.size(2))).long()], dim=1)

            pred = pred.view(-1, vocab_size)
            batchY = batchY.view(-1)
            
            loss = CE_loss_fn(pred, batchY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.detach().numpy())

        epoch_loss = np.mean(batch_losses)
        epoch_er = evaluate(val_dataset, model, device)
        logging.info(f'completed epoch: {epoch_idx + 1}, average loss: {epoch_loss}, error rate: {epoch_er}')

        if epoch_loss < best_epoch_loss:
            torch.save(model.state_dict, os.path.join(FLAGS.model_folder, 'best_loss_model.pt'))
        if epoch_er > best_epoch_acc:
            torch.save(model.state_dict, os.path.join(FLAGS.model_folder, 'best_er_model.pt'))

        torch.save(model.state_dict, os.path.join(FLAGS.model_folder, 'model.pt'))

    return model



def main():

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info('process started')

    if torch.cuda.is_available():
        logging.info('CUDA device available')
        device = 'cuda'
    else:
        logging.info('CUDA device NOT available')
        device = 'cpu'

    if FLAGS.tokenizer == 'char':
        tokenizer = Tokenizer(type='char')
    else:
        tokenizer = Tokenizer(type='subword')
    
    logging.info('Tokenizer setup complete')

    if FLAGS.evaluate_saved is not None:
        testset = MotionDataset(tokenizer, test=True)
        model = torch.load(os.path.join('models/', FLAGS.evaluate_saved))
        err_rate = evaluate(testset, model, device)

        logging.info('evaluation complete')
    else:
        tokenizer.create_vocab_file(FLAGS.vocab_file)
        trainset = MotionDataset(tokenizer)
        valset = MotionDataset(tokenizer, val=True)
        model = train(trainset, valset, device)

        logging.info('training complete')
    

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()

