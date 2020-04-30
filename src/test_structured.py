import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import math
from utils import *
from train_structured import *
from evaluate_structured import *
from config import model_config as config
from allennlp.modules.elmo import batch_to_ids

class Dataset(data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, elmo_pair, x_pair, t_pair, d_pair):
        #'Initialization'
        if config['elmo']:
            self.elmo_train = elmo_pair
        self.x_train = x_pair
        self.t_pair = t_pair
        self.d_pair = d_pair

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.x_train)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample

        # Load data and get label
        elmo = []
        if config['elmo']:
            elmo = self.elmo_train[index]
        x = self.x_train[index]
        t = self.t_pair[index]
        d = self.d_pair[index]

        return elmo, x,t,d

def load_data(dataset):
    dataloader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    return dataloader

def trainIters(decoder, pairs, valid_pairs, test_pairs, output_lang, tag_lang, dep_lang, train_pos, train_dep, valid_pos, valid_dep, test_pos, test_dep):
    start = time.time()
    plot_losses = []
    loss_final = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.ASGD(decoder.parameters())#, lr=config['lr'])
    # TODO: use better optimizer
    x_pair = []
    d_pair = []
    t_pair = []
    y_pair = []
    elmo_pair = []
    if config['elmo']:
        for i in range(len(pairs)):
            elmo_pair.append(pairs[i].split())
        elmo_pair = batch_to_ids(elmo_pair)
        #print(elmo_pair.shape)
    for i in range(len(pairs)):
        x_pair.append(tensorsFromPair(pairs[i], output_lang))
        d_pair.append(tensorsFromPair(train_dep[i], dep_lang))
        t_pair.append(tensorsFromPair(train_pos[i], tag_lang))
        #x_pair.append(pair[0])
        #y_pair.append(pair[1])
    '''print(x_pair)
    print('haha')
    print(d_pair)
    print('asdf')
    print(t_pair)'''
    #print(elmo_pair)
    if config['elmo']:
        elmo_pair = [pad_sequences(x, config['MAX_LENGTH'], config['elmo']) for x in elmo_pair]
    #print(elmo_pair)
    #print(x_pair)
    x_pair = [pad_sequences(x, config['MAX_LENGTH'], False) for x in x_pair]
    #print(x_pair)
    d_pair = [pad_sequences(x, config['MAX_LENGTH'], False) for x in d_pair]
    t_pair = [pad_sequences(x, config['MAX_LENGTH'], False) for x in t_pair]
    #print(elmo_pair[0].shape)
    #print(x_pair[0].shape)
    #print(t_pair[0].shape)
    #y_pair = [pad_sequences(x, MAX_LENGTH) for x in y_pair]
    print('Dataset prepared, preparing iterator')
    training_set = Dataset(elmo_pair, x_pair, t_pair, d_pair)
    training_iterator = load_data(training_set)
    perp = 10000
    criterion = nn.NLLLoss(ignore_index=0)
    for epoch in range(config['epochs']):
        print('epoch')
        print(epoch+1)
        i = 0
        loss_final = 0
        for elmo_tensor, target_tensor, tag_tensor, dep_tensor in training_iterator:
            '''if config['elmo']:
                elmo_tensor = elmo_tensor.to(device)
            target_tensor = target_tensor.to(device)
            tag_tensor = tag_tensor.to(device)
            dep_tensor = dep_tensor.to(device)'''
            #print(target_tensor.shape)
            #print(elmo_tensor.shape)
            target_tensor_len = get_len(target_tensor)
            #target_tensor_mask = get_mask(target_tensor_len)
            tag_tensor_len = get_len(tag_tensor)
            #print(tag_tensor_len)
            #tag_tensor_mask = get_mask(tag_tensor_len)
            dep_tensor_len = get_len(dep_tensor)
            #dep_tensor_mask = get_mask(tag_tensor_len)
            loss = train(elmo_tensor, target_tensor, target_tensor_len, tag_tensor, tag_tensor_len,
                     dep_tensor, dep_tensor_len, decoder, decoder_optimizer, criterion)
            i += 1
            print_loss_total += loss
            plot_loss_total += loss
            loss_final += loss
            #print(loss)
            if i % config['print_every'] == 0:
                #print(print_loss_total)
                print_loss_avg = print_loss_total / config['print_every']
                print_loss_total = 0
                iters = epoch*len(pairs) + (config['batch_size']*i)
                n_iters = config['epochs']*len(pairs)
                #print(loss)
                print('%s (%d%%) %.4f %.4f' % (timeSince(start, iters/n_iters),
                                             iters / n_iters * 100, print_loss_avg, math.exp(loss_final/i)))
        #print("final loss")
        #print(loss_final / i)
        print("perplexity")
        print(math.exp(loss_final/i))
        

        #print("Perplexity")
        #print("")
        #showPlot(plot_losses)
        #evaluateBLUE(decoder, pairs, input_lang, output_lang, False)
        ep = evaluatePerplexity(decoder, valid_pairs, output_lang, tag_lang, dep_lang, False)
        if (perp > ep):
            torch.save(decoder.state_dict(), config['lm_name'] + '.pt')
            decoder.load_state_dict(torch.load(config['lm_name'] + '.pt'))
            perp = ep
            print('language model saved')
        evaluatePerplexity(decoder, test_pairs, output_lang, tag_lang, dep_lang, False)
