import torch
from config import model_config as config
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model.weight_drop import WeightDrop
from model.embed_regularize import embedded_dropout
from model.locked_dropout import LockedDropout

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, embedding_weights, embedding_dim, dropout_p):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.lockdrop = LockedDropout()
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_i = nn.Dropout(0.65)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers = self.num_layers)
        '''self.layers = []
        for i in range(self.num_layers):
            self.layers.append('weight_hh_l'+str(i))
        self.gru = WeightDrop(self.gru, self.layers, dropout=self.wdrop)'''
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, training):
        #embedded = embedded_dropout(self.embedding, input, dropout=self.dropout_e if training else 0)
        #embedded = self.lockdrop(embedded, self.dropouti)
        embedded = self.embedding(input).permute(1, 0, 2)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        #output = self.dropout(output)
        output = self.softmax(self.out(output[0]))
        #print(output.shape)
        return output.unsqueeze(0), hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
