import os
import sys
import tqdm
import random
import numpy as np 

import torch
import torch.nn as nn
import torch.functional as F  

import transformers

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased')

class LSTMSentenceEncoding(nn.Module):
	'''Create an LSTM sentence encoding using the lemma IDs passed in'''
	def __init__(self, num_words, num_layers, hidden_size, dropout = 0):
		super(LSTMSentenceEncoding, self).__init__()
		#Pass through initial embedding layer
		self.embedding = nn.Embedding(num_words, hidden_size)
		#Use LSTM with num_layers number of layers, setting bidirectional to true
		self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout, batch_first = True, bidirectional = True)

	def forward(self, x):
		x = self.embedding(x)
		outputs, _ = self.lstm(x)
		return outputs

class LSTMClassifier1(nn.Module):
	def __init__(self, num_words, num_labels, num_layers, hidden_size, dropout = 0):
		'''LSTM Sentence encoding with 9 linear layers'''
		super(LSTMClassifier1, self).__init__()
		self.sent_enc = LSTMSentenceEncoding(num_words, num_layers, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.linear1 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear2 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear3 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear4 = nn.Linear(2*hidden_size, hidden_size)
		self.linear5 = nn.Linear(2*hidden_size, hidden_size)
		self.linear6 = nn.Linear(2*hidden_size, hidden_size)
		self.linear7 = nn.Linear(2*hidden_size, hidden_size)
		self.linear8 = nn.Linear(2*hidden_size, hidden_size)
		self.linear9 = nn.Linear(hidden_size, num_labels)

	def forward(self, x):
		x = self.sent_enc(x)
		x.squeeze(0)
		x = self.dropout(x)
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.linear3(x)
		x = self.linear4(x)
		x = self.linear5(x)
		x = self.linear6(x)
		x = self.linear7(x)
		x = self.linear8(x)
		logits = self.linear9(x)
		logits = logits.squeeze(0)
		return logits

class LSTMClassifier2(nn.Module):
	'''LSTM Sentence Encoding with LSTM layer and 9 linear layers'''
	def __init__(self, num_words, num_labels, num_layers, hidden_size, dropout = 0):
		super(LSTMClassifier2, self).__init__()
		self.sent_enc = LSTMSentenceEncoding(num_words, num_layers, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(2*hidden_size, hidden_size, num_layers = num_layers, dropout = dropout, batch_first = True, bidirectional = True)
		self.linear1 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear2 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear3 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear4 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear5 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear6 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear7 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear8 = nn.Linear(2*hidden_size, hidden_size)
		self.linear9 = nn.Linear(hidden_size, num_labels)

	def forward(self, x):
		x = self.sent_enc(x)
		x = self.dropout(x)
		x, _ = self.lstm(x)
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.linear3(x)
		x = self.linear4(x)
		x = self.linear5(x)
		x = self.linear6(x)
		x = self.linear7(x)
		x = self.linear8(x)
		logits = self.linear9(x)
		logits = logits.squeeze(0)
		return logits

class LMClassifier1(nn.Module):
	'''LM/BERT sentence encoding with linear layers'''
	def __init__(self, num_labels, hidden_size, lm = None, dropout = 0.2):
		super(LMClassifier1, self).__init__()
		self.lm = lm
		self.dropout = nn.Dropout(dropout)
		self.linear1 = nn.Linear(lm.config.hidden_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.linear4 = nn.Linear(hidden_size, hidden_size)
		self.linear5 = nn.Linear(hidden_size, hidden_size)
		self.linear6 = nn.Linear(hidden_size, hidden_size)
		self.linear7 = nn.Linear(hidden_size, hidden_size)
		self.linear8 = nn.Linear(hidden_size, hidden_size)
		self.linear9 = nn.Linear(hidden_size, num_labels)

	def forward(self, input_ids = None, attention_mask = None):
		lm_output = self.lm(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
		hidden_states = lm_output.hidden_states[7]
		hidden_states = self.dropout(hidden_states)
		x = self.linear1(hidden_states)
		x = self.linear2(x)
		x = self.linear3(x)
		x = self.linear4(x)
		x = self.linear5(x)
		x = self.linear6(x)
		x = self.linear7(x)
		x = self.linear8(x)
		logits = self.linear9(x)
		logits = logits.squeeze(0)
		logits = logits[1:-1, :]
		return logits

class LMClassifier2(nn.Module):
	'''BERT embeddings with LSTM layer and 9 linear layers. Most complex model.
	'''
	def __init__(self, hidden_size, num_layers, num_labels, lm = None, dropout = 0.2):
		super(LMClassifier2, self).__init__()
		self.lm = lm
		self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(lm.config.hidden_size, hidden_size, num_layers = num_layers, dropout = dropout, batch_first = True, bidirectional = True)
		self.linear1 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear2 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear3 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear4 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear5 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear6 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear7 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear8 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.linear9 = nn.Linear(2*hidden_size, num_labels)

	def forward(self, input_ids = None, attention_mask = None):
		lm_output = self.lm(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
		hidden_states = lm_output.hidden_states[7]
		hidden_states = self.dropout(hidden_states)
		lstm_output, _ = self.lstm(hidden_states)
		x = self.linear1(lstm_output)
		x = self.linear2(x)
		x = self.linear3(x)
		x = self.linear4(x)
		x = self.linear5(x)
		x = self.linear6(x)
		x = self.linear7(x)
		x = self.linear8(x)
		logits = self.linear9(x)
		logits = logits.squeeze(0)
		logits = logits[1:-1, :]
		return logits