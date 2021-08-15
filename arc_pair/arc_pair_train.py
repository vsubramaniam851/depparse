import sys
import os
import random
import numpy as np 

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim

from arc_pair_model import *
from data_loader import *

import transformers

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

base_path = '/storage/vsub851/depparse/UD_English-EWT'
filename = 'small.conllu'

print('Using device: {}'.format(device)) #Ensure on GPU!

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def arc_train(train_corpus, train_type, num_words, num_labels, modelname, base_path, hidden_size = 200, lr = 0.005, dropout = 0.25, num_epochs = 3, num_layers = 2, batch_size = 1, model = 'LSTM1', lm = None):
	'''Train the model. Specify the model type and hyperparameters. The LSTM model takes in lemmas in a sentence and predicts its heads and dependencies
	The LM model uses the input ids and attention masks instead. 

	Pass in the training type, either on heads or dependencies since we train each separately
	'''
	#Choose model based on model parameter.
	if model == 'LSTM1':
		classifier = LSTMClassifier1(num_words, num_labels, num_layers, hidden_size, dropout)
	elif model == 'LSTM2':
		classifier = LSTMClassifier2(num_words, num_labels, num_layers, hidden_size, dropout)
	elif model == 'LM1':
		classifier = LMClassifier1(num_labels = num_labels, hidden_size = hidden_size, lm = lm, dropout = dropout)
	else:
		classifier = LMClassifier2(num_labels = num_labels, hidden_size = hidden_size, lm = lm, dropout = dropout, num_layers = num_layers)
	optimizer = optim.Adam(classifier.parameters(), lr = lr)
	#CLassification problem using Cross Entropy Loss
	loss_fn = nn.CrossEntropyLoss().to(device)

	classifier = classifier.to(device)
	classifier.train()

	#Collect correct label type
	if train_type == 'deprel':
		label_type = 'pair_deprel'
	else:
		label_type = 'pair_heads'

	#Training loop
	print('Beginning training on {}'.format(train_type))
	for epoch in range(num_epochs):
		total_loss = 0
		classifier.train()
		for i in range(1, len(train_corpus), batch_size):
			if i % 1000 == 0:
				print('Epoch {} Batch {}'.format(epoch, i))
			batch = train_corpus[i:i+batch_size]
			word_batch = []
			start_pairs = []
			end_pairs = []
			input_ids = []
			attention_mask = []
			labels_batch = []
			for sent in batch:
				#Uncomment corresponding lines depending on whether LSTM is being used or LM
				word_batch.append(torch.tensor(sent['lemma_ids']).long().to(device))
				start_pairs.append(torch.tensor(sent['start_arc']).long().to(device))
				end_pairs.append(torch.tensor(sent['end_arc']).long().to(device))
				# input_ids.append(torch.tensor(sent['input_ids']).long().to(device))
				# attention_mask.append(torch.tensor(sent['attention_mask']).long().to(device))
				labels1 = torch.tensor(sent[label_type]).long().to(device)
				labels_batch.append(labels1)
			word_batch = torch.stack(word_batch).to(device)
			start_pairs = torch.stack(start_pairs).to(device)
			start_pairs = start_pairs.squeeze(0)
			end_pairs = torch.stack(end_pairs).to(device)
			end_pairs = end_pairs.squeeze(0)
			# input_ids = torch.stack(input_ids).to(device)
			# attention_mask = torch.stack(attention_mask).to(device)
			labels_batch = torch.stack(labels_batch).to(device)
			labels_batch = labels_batch.squeeze(0)

			#Run LSTM
			outputs = classifier.forward(word_batch, start_pairs, end_pairs)
			#Run LM
			# outputs = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, start_pairs = start_pairs, end_pairs = end_pairs)

			#Calculate loss and step backwards through the model.
			loss = loss_fn(outputs, labels_batch)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_corpus)))
	print('TRAINING IS FINISHED')
	#Save model using modelname passed in as parameter
	save_path = os.path.join(base_path, 'checkpoints')
	if os.path.exists(save_path):
		torch.save(classifier.state_dict(), os.path.join(save_path, modelname))
	else:
		print('Give a valid directory and path!')

#TEST CODE for checking whether the training function is working separately
def test_train(base_path, model_name, filename = 'en_ewt-ud-train.conllu', lm = None):
	# Hyperparameters of the model which can be tuned 
	batch_size = 1
	num_layers = 2
	hidden_size = 200
	lr = 0.00005
	num_epochs = 1
	dropout = 0.33
	model = 'LM1'
	lm = lm

	# Loading the data from the file
	print('Starting data processing')
	lines = preproc_conllu(base_path, filename = 'en_ewt-ud-train.conllu')
	sent_collection = sentence_collection(lines)
	train_corpus, vocab_dict, label_dict = process_corpus(sent_collection, mode = 'train')
	train_corpus = bert_tokenizer(train_corpus)
	train_corpus = form_pairs(train_corpus)
	num_words = len(vocab_dict)
	num_labels = len(label_dict)
	print('Finished data processing')

	#Training code. Tune hyperparameters are necessary.
	arc_train(train_corpus, train_type = 'deprel', num_words = num_words, num_labels = num_labels, modelname = model_name, hidden_size = hidden_size, lr = lr, dropout = dropout, num_epochs = num_epochs, 
			num_layers = num_layers, batch_size = batch_size, model = model, lm = lm)

# test_train(base_path = base_path, model_name = 'lm1deprel.pt', lm = lm_pretrained)