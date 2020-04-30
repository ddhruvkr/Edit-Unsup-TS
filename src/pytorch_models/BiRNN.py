import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class BiRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.fc = nn.Linear(hidden_dim*8, output_dim)
		self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)
		self.hidden_dim = hidden_dim
		self.dropout = nn.Dropout(dropout_prob)
		self.avg_pool = True
		self.softmax = nn.LogSoftmax(dim=1)
		self.Wy = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
		self.Wh = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
		self.w = nn.Linear(2*hidden_dim, 1, bias=False)

	def forward(self, inp1, inp2):
		embeddings1 = self.embedding(inp1)
		embeddings2 = self.embedding(inp2)
		#batch_size, max_sequence_length, embedding_length
		embeddings1 = embeddings1.permute(1, 0, 2)
		embeddings2 = embeddings2.permute(1, 0, 2)
		output1, hidden1 = self.rnn(embeddings1)
		output2, hidden2 = self.rnn(embeddings2)
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]
		x1 = output1.permute(1,2,0)
		x2 = output2.permute(1,2,0)
		#batch_size, 2*hidden_dim, max_sequence_length
		if self.avg_pool == True:
			x1 = f.avg_pool1d(x1, x1.size(2))
			x2 = f.avg_pool1d(x1, x2.size(2))
		else:
			x1 = f.relu(f.max_pool1d(x1, x.size(2)))
			x2 = f.relu(f.max_pool1d(x2, x.size(2)))
		#(batch_size, 2*hidden_dim, 1)
		x1 = x1.squeeze(2)
		x2 = x2.squeeze(2)
		x1 = self.attention_layer(output1.permute(1,0,2),x1)
		x2 = self.attention_layer(output2.permute(1,0,2),x2)
		x3 = x1*x2
		x4 = torch.abs(x1-x2)
		x = torch.cat((x1,x3),dim=1)
		x = torch.cat((x,x4),dim=1)
		x = torch.cat((x,x2),dim=1)
		linear=self.fc(x)
		x = self.dropout(x)
		#linear = self.softmax(linear)
		return linear


	def attention_layer(self, output_bilstm, hidden_pool):
		#print('output bilstm')
		#print(output_bilstm.shape)
		a = self.Wy(output_bilstm)
		#print('a')
		#print(a.shape)
		#(2d, 2d)(bts, n, 2d) = (bts, n, 2d)
		x = self.Wh(hidden_pool)
		#print('x')
		#print(x.shape)
		#(2d,2d)(bts,2d) = (bts,2d)
		x = x.unsqueeze(2)
		#print(x.shape)
		#(bts, 2d, 1)
		x = x.repeat(1,1,output_bilstm.shape[1])
		#print(x.shape)
		#(bts, 2d, n)
		x = x.permute(0,2,1)
		#print(x.shape)
		M = torch.add(a,x)
		#print(M.shape)
		M = torch.tanh(M)
		att = self.w(M)
		#print('att')
		#(2d,1)(bts,n,2d) = (bts, n, 1)
		#print(att.shape)
		att = f.softmax(att, dim=1) #along n
		#print(att.shape)
		att = att.permute(0,2,1)
		Rattn = torch.bmm(att, output_bilstm)
		#print(Rattn.shape)
		#(bts, 1, n)(bts, n, 2d) = (bts, 1, 2d)
		Rattn = Rattn.squeeze(1)
		return Rattn
