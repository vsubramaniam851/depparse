import sys
import os
import random
import numpy as np 

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim

from arc_pred_model import *
from data_loader import *

MODEL_FOLDER = '/storage/vsub851/864_final_project/checkpoints'

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

def decode_model(outputs, decode_type = 'deprel', sent_len = 0):
	'''Decode the outputs from the model, each output should be of size [sentence_len, num_labels]. 
	The decode function finds the position with the largest probability and returns it as the most likely label,
	applies to both heads and dependencies
	'''
	preds = []
	#Iterate over the outputs on each probability distribution
	for i in range(len(outputs)):
		seq = outputs[i]
		#If it is heads, then the num_labels will be the length of the longest sentence in the data. Slice to get our sentence length
		if decode_type == 'heads':
			seq = seq[:sent_len + 1]
		preds.append(torch.argmax(seq))
	return preds

def eval_model(test_corpus, num_words,model_state_dict1, model_state_dict2 = None, num_heads = None, num_deprel = None, model = 'LSTM1', dropout = 0.25, num_layers = 2, hidden_size = 200, test_type = 'UAS', lm = None):
	'''Evaluate the model on the test corpus to see that the accuracy is. Pass in a model state dictionary as a pytorch file which has the model saved.
	eval_type gives the type of evaluation. UAS just evaluates heads while LAS evaluates heads and labels so you need both model states to be passed 
	in to run both head and dependency model
	'''

	#Head eval
	if test_type == 'UAS':
		#Select the model type
		if model == 'LSTM1':
			classifier = LSTMClassifier1(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
		elif model == 'LSTM2':
			classifier = LSTMClassifier2(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
		elif model == 'LM1':
			classifier = LMClassifier1(num_labels = num_heads, hidden_size = hidden_size, lm = lm, dropout = dropout)
		else:
			classifier = LMClassifier2(num_labels = num_heads, hidden_size = hidden_size, lm = lm, num_layers = num_layers, dropout = dropout)
		model_loc = os.path.join(MODEL_FOLDER, model_state_dict1)
		classifier.load_state_dict(torch.load(model_loc))
		classifier = classifier.to(device)

		label_type = 'heads'
		classifier.eval()
		total_examples = 0
		total_correct = 0

		for i in range(len(test_corpus)):
			sent_dict = test_corpus[i]
			word_batch = []
			sent_len = len(sent_dict['lemma_ids'])
			# word_batch.append(torch.tensor(sent_dict['lemma_ids']).long().to(device))
			# word_batch = torch.stack(word_batch).to(device)
			input_ids = []
			attention_mask = []
			#Collect right inputs for right model
			input_ids.append(torch.tensor(sent_dict['input_ids']).long().to(device))
			attention_mask.append(torch.tensor(sent_dict['attention_mask']).long().to(device))
			input_ids = torch.stack(input_ids).to(device)
			attention_mask = torch.stack(attention_mask).to(device)
			labels = torch.tensor(sent_dict[label_type]).long().to(device)

			#Run and decode classifier
			outputs = classifier.forward(input_ids = input_ids, attention_mask = attention_mask)
			preds = decode_model(outputs, decode_type = 'heads', sent_len = sent_len)

			#Count correct versus total examples
			for i in range(len(preds)):
				if preds[i] == labels[i]:
					total_correct += 1
				total_examples += 1
		return ('UAS score: {}').format(total_correct/total_examples)

	else:
		#Check that we have TWO saved models that are being evaluated
		if model_state_dict1 == None or model_state_dict2 == None:
			return 'Please give a saved model'

		#Select the model we are evaluating
		if model == 'LSTM1':
			head_classifier = LSTMClassifier1(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
			dep_classifier = LSTMClassifier1(num_words, num_deprel, num_layers, hidden_size, dropout = dropout)
		elif model == 'LSTM2':
			head_classifier = LSTMClassifier2(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
			dep_classifier = LSTMClassifier2(num_words, num_deprel, num_layers, hidden_size, dropout = dropout)
		elif model == 'LM1':
			head_classifier = LMClassifier1(num_labels = num_heads, hidden_size = hidden_size, lm = lm, dropout = dropout)
			dep_classifier = LMClassifier1(num_labels = num_deprel, hidden_size = hidden_size, lm = lm, dropout = dropout)
		else:
			head_classifier = LMClassifier2(num_labels = num_heads, hidden_size = hidden_size, lm = lm, num_layers = num_layers, dropout = dropout)
			dep_classifier = LMClassifier2(num_labels = num_deprel, hidden_size = hidden_size, lm = lm, num_layers = num_layers, dropout = dropout)

		#Get both head and dependency model
		head_model_loc = os.path.join(MODEL_FOLDER, model_state_dict1)
		deprel_model_loc = os.path.join(MODEL_FOLDER, model_state_dict2)
		head_classifier.load_state_dict(torch.load(head_model_loc))
		dep_classifier.load_state_dict(torch.load(deprel_model_loc))

		#Set both classifiers to eval
		head_classifier = head_classifier.to(device)
		dep_classifier = dep_classifier.to(device)

		head_classifier.eval()
		dep_classifier.eval()
		total_examples = 0
		total_correct = 0
		
		for i in range(len(test_corpus)):
			sent_dict = test_corpus[i]
			word_batch = []
			sent_len = len(sent_dict['lemma_ids'])
			word_batch.append(torch.tensor(sent_dict['lemma_ids']).long().to(device))
			word_batch = torch.stack(word_batch).to(device)
			heads = torch.tensor(sent_dict['heads']).long().to(device)
			deprels = torch.tensor(sent_dict['deprel_ids']).long().to(device)

			#Run both models
			output_heads = head_classifier.forward(word_batch)
			output_deps = dep_classifier.forward(word_batch)
			pred_heads = decode_model(output_heads, decode_type = 'heads', sent_len = sent_len)
			pred_deps = decode_model(output_deps)

			#Count whether both heads and dependencies match
			for i in range(len(pred_heads)):
				if pred_heads[i] == heads[i] and pred_deps[i] == deprels[i]:
					total_correct += 1
				total_examples += 1
		return ('LAS score: {}').format(total_correct/total_examples)

#TEST CODE FOR TESTING THE EVAL FUNCTION
def test_eval(test_corpus, num_words,model_state_dict1, model_state_dict2 = None, num_heads = None, num_deprel = None, model = 'LSTM1', dropout = 0.25, num_layers = 2, hidden_size = 200, test_type = 'UAS', lm = None):
	eval_model(test_corpus, num_words,model_state_dict1, model_state_dict2, num_heads, num_deprel, model, dropout, num_layers, hidden_size, test_type, lm)