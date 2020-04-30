import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import *
import math
from config import model_config as config

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train(target_tensor, target_tensor_mask, target_tensor_len, decoder, decoder_optimizer, criterion):
    decoder_hidden = decoder.initHidden(config['batch_size'])
    #decoder_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    #encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #print(input_tensor)
    #input_length = input_tensor.size(1)
    #target_length = max_length
    target_length = torch.max(target_tensor_len).item()
    loss = 0
    #print(target_tensor.shape)
    decoder_input = target_tensor.narrow(1, 0, 1)
    # both decoder_input definitions should work
    #decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
    for di in range(target_length-1):
        #print(di)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, True)
        indices = torch.tensor([di+1], device=device)
        #print("indices")
        #print(indices)
        target_t = torch.index_select(target_tensor, 1, indices)
        loss += criterion(decoder_output[0], target_t.view(-1))
        decoder_input = target_t
        #if decoder_input.item() == 0:
        #    break 
    loss.backward()


    # because this is happening time step by time step for multiple batched
    # and by default losses are averaged over all elements
    # the loss we get is averaged over number of batches
    # below we divide by total number of words in a sentence
    # but ideally we should divide by number of non-pad words in a sentence since 
    # because of ignore index, pad words are not considered while calculating loss
    # but here we don't mind it and use max_length anyways
    # however, in evaluate we will keep this in mind since we want accurate perplexity values there
    #print(loss)
    #clip_gradient(encoder, 100e-1)
    clip_gradient(decoder, 100e-1)
    #encoder_optimizer.step()
    decoder_optimizer.step()
    final_loss = loss.item() / target_length
    return final_loss
