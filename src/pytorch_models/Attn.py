import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class WordAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.it = 60 # should be tuned
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.bilstms = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.Ww = nn.Linear(2*hidden_dim, self.it)
		self.Uw = nn.Linear(self.it, 1, bias=False)
		self.label = nn.Linear(hidden_dim*4, output_dim)
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		self.hidden_dim = hidden_dim
		self.add_pool = True
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input_sequences):
		#shape of x:
		#batch_size, max_sequence_length
		embeddings = self.embedding(input_sequences)
		embeddings = embeddings.permute(1, 0, 2)
		#max_sequence_length, batch_size, embedding_length
		output, (hidden, cell) = self.bilstms(embeddings)
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]
		if self.add_pool == True:
			x = output.permute(1,2,0)
			#batch_size, 2*hidden_dim, max_sequence_length
			x = f.max_pool1d(x, x.size(2))
			#(batch_size, 2*hidden_dim, 1)
			x = x.squeeze(2)
		output = output.permute(1,0,2)
		s = self.attention_layer(output)
		s = torch.cat((s, x), dim=1)
		#batch_size, 2d
		linear = self.label(s)
		linear = self.softmax(linear)
		#print(linear.shape)
		return linear

	def attention_layer(self, output_bilstm):
		u = self.Ww(output_bilstm)
		#(2d, it)(bts, n, 2d) = (bts, n, it)
		u_tanh = torch.tanh(u)
		#u_tanh = f.relu(u)
		att = self.Uw(u_tanh)
		#(it,1)(bts,n,it) = (bts, n, 1)
		att = att.permute(0,2,1)
		#(bts, 1, n)
		att = f.softmax(att, dim=2) #along n
		att = att.squeeze(1)
		#bts,n
		output_bilstm = output_bilstm.permute(2,0,1)
		#2*hidden_dim, bts,n
		si = torch.mul(att, output_bilstm)
		#2d, bts, n
		si = si.transpose(0,1)
		#bts,2d,n
		si = torch.sum(si, dim=2)
		#bts,2d
		return si
