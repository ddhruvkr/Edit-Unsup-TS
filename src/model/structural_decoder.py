import torch
from config import model_config as config
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from model.weight_drop import WeightDrop
from model.embed_regularize import embedded_dropout
from model.locked_dropout import LockedDropout
device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")
#options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
#weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
#elmo = Elmo(options_file, weight_file, 1,requires_grad=False, dropout=0.5,device=device)

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, tag_size, dep_size, num_layers, embedding_weights, tag_embedding_weights, dep_embedding_weights, embedding_dim, tag_dim, dep_dim, dropout_p, use_structural_as_standard):
        super(DecoderGRU, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tag_size = tag_size
        self.dep_size = dep_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.use_structural_as_standard = use_structural_as_standard
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        if not self.use_structural_as_standard:
            self.tag_embedding = nn.Embedding(self.tag_size, tag_dim)
            self.dep_embedding = nn.Embedding(self.dep_size, dep_dim)
        self.dropout = nn.Dropout(dropout_p)
        if config['elmo']:
            self.options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
            self.weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
            self.elmo = Elmo(self.options_file, self.weight_file, 1,requires_grad=True, dropout=0.5)
            self.gru = nn.GRU(2*self.embedding_dim+tag_dim+dep_dim, self.hidden_size, num_layers = self.num_layers)
        else:
            if self.use_structural_as_standard:
                self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers = self.num_layers)
            else:
                self.gru = nn.GRU(self.embedding_dim+tag_dim+dep_dim, self.hidden_size, num_layers = self.num_layers)
        #self.gru = nn.GRU(3*self.embedding_dim, self.hidden_size, num_layers = self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
        #self.tag_embedding.weight = nn.Parameter(torch.from_numpy(tag_embedding_weights).float(), requires_grad=True)
        #self.dep_embedding.weight = nn.Parameter(torch.from_numpy(dep_embedding_weights).float(), requires_grad=True)
        self.softmax = nn.LogSoftmax(dim=1)


        if config['awd']:
            self.lockdrop = LockedDropout()
            self.dropout_e = 0.1
            self.dropout_h = 0.3
            self.wdrop = 0.3
            #self.dropout = nn.Dropout(0.4)
            self.dropout_i = nn.Dropout(0.3)
            self.layers = []
            for i in range(self.num_layers):
                self.layers.append('weight_hh_l'+str(i))
            self.gru = WeightDrop(self.gru, self.layers, dropout=self.wdrop)

    def forward(self, input, tag_input, dep_input, hidden, training):

        if config['awd']:
            embedded = embedded_dropout(self.embedding, input, dropout=self.dropout_e if training else 0)
            #embedded = self.lockdrop(embedded, self.dropout_i)
            #embedded = self.embedding(input).permute(1, 0, 2)
            #embedded = self.dropout_i(embedded)
            #output, hidden = self.gru(embedded, hidden)
            #output = self.dropout(output)
            #output = self.softmax(self.out(output[0]))
            #print(output.shape)
            #return output.unsqueeze(0), hidden

        if config['elmo']:
            embeddings_elmo = self.elmo(input)
            embeddings_elmo = embeddings_elmo['elmo_representations'][0]
            embedded = embeddings_elmo.permute(1, 0, 2)
        else:
            embedded = self.embedding(input).permute(1, 0, 2)
        if not self.use_structural_as_standard:
            tag_embedded = self.tag_embedding(tag_input).permute(1, 0, 2)
            dep_embedded = self.dep_embedding(dep_input).permute(1, 0, 2)
            embedded = torch.cat([embedded,tag_embedded,dep_embedded],dim=2)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)
        output = self.softmax(self.out(output[0]))
        #print(output.shape)
        return output.unsqueeze(0), hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
