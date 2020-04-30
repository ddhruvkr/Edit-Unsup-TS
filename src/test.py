import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import math
from utils import *
from train import *
from evaluate import *
from config import model_config as config

def trainIters(decoder, pairs, valid_pairs, test_pairs, output_lang):
    start = time.time()
    plot_losses = []
    loss_final = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.ASGD(decoder.parameters())#, lr=config['lr'])
    # TODO: use better optimizer
    x_pair = []
    y_pair = []
    for i in range(len(pairs)):
        x_pair.append(tensorsFromPair(pairs[i], output_lang))
        #doc=nlp(pairs[i])
        #d_pair.append(tensorsFromPair(convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang))
        #print(output_lang.n_words)
        #x_pair.append(tensorsFromPair(convert_to_sent([(tok.tag_).upper() for tok in doc]), output_lang))
        #x_pair.append(pair[0])
        #y_pair.append(pair[1])
    '''print(x_pair)
    print('haha')
    print(d_pair)
    print('asdf')
    print(t_pair)'''
    x_pair = [pad_sequences(x, config['MAX_LENGTH']) for x in x_pair]
    #y_pair = [pad_sequences(x, MAX_LENGTH) for x in y_pair]
    print('Dataset prepared, preparing iterator')
    training_set = Dataset(x_pair)
    training_iterator = load_data(training_set)
    perp = 10000
    criterion = nn.NLLLoss(ignore_index=0)
    for epoch in range(config['epochs']):
        print('epoch')
        print(epoch+1)
        i = 0
        loss_final = 0
        for target_tensor in training_iterator:
            target_tensor_len = get_len(target_tensor)
            target_tensor_mask = get_mask(target_tensor_len)
            loss = train(target_tensor, target_tensor_mask, target_tensor_len, decoder, decoder_optimizer, criterion)
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
        ep = evaluatePerplexity(decoder, valid_pairs, output_lang, False)
        if (perp > ep):
            torch.save(decoder.state_dict(), config['lm_name'] + '.pt')
            decoder.load_state_dict(torch.load(config['lm_name'] + '.pt'))
            perp = ep
            print('language model saved')
        evaluatePerplexity(decoder, test_pairs, output_lang, False)
