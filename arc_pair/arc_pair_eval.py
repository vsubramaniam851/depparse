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

MODEL_FOLDER = '/storage/vsub851/depparse'

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

def decode_model(outputs):
	'''Decode the outputs from the model, each output should be of size [sentence_len, num_labels]. 
	The decode function finds the position with the largest probability and returns it as the most likely label,
	applies to both heads and dependencies
	'''
	preds = []
	for dist in outputs:
		preds.append(torch.argmax(dist))
	return preds

def eval_model(test_corpus, num_words,model_state_dict1, model_state_dict2 = None, base_path ='/storage/vsub851/depparse', num_heads = 2, num_deprel = None, model = 'LSTM1', dropout = 0.25, num_layers = 2, hidden_size = 200, test_type = 'UAS', lm = None):
	'''Evaluate the model on the test corpus to see that the accuracy is. Pass in a model state dictionary as a pytorch file which has the model saved.
	eval_type gives the type of evaluation. UAS just evaluates heads while LAS evaluates heads and labels so you need both model states to be passed 
	in to run both head and dependency model
	'''

	#Head eval
	if test_type == 'UAS':
		print('Beginning UAS Eval')

		#Select the model type
		if model == 'LSTM1':
			classifier = LSTMClassifier1(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
		elif model == 'LSTM2':
			classifier = LSTMClassifier2(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
		elif model == 'LM1':
			classifier = LMClassifier1(num_labels = num_heads, hidden_size = hidden_size, lm = lm, dropout = dropout)
		else:
			classifier = LMClassifier2(num_labels = num_heads, hidden_size = hidden_size, lm = lm, num_layers = num_layers, dropout = dropout)
		model_folder = os.path.join(base_path, 'arc_pair', 'checkpoints')
		model_loc = os.path.join(model_folder, model_state_dict1)
		classifier.load_state_dict(torch.load(model_loc))
		classifier = classifier.to(device)

		label_type = 'pair_heads'
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
			start_pairs = torch.tensor(sent_dict['start_arc']).long().to(device)
			end_pairs = torch.tensor(sent_dict['end_arc']).long().to(device)

			#Run and decode classifier
			#LSTM
			# outputs = classifier.forward(word_batch, start_pairs, end_pairs)
			#LM
			outputs = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, start_pairs = start_pairs, end_pairs = end_pairs)
			#Decode model to get predictions of which pairs are head-dependent
			preds = decode_model(outputs)

			#Count correct versus total examples
			for i in range(len(preds)):
				if preds[i] == labels[i]:
					total_correct += 1
				total_examples += 1
		return ('UAS score: {}').format(total_correct/total_examples)
	#Head and Deprel Eval
	else:
		print('Beginning {} evaluation').format(test_type)

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
		model_folder = os.path.join(base_path, 'arc_pair', 'checkpoints')
		head_model_loc = os.path.join(model_folder, model_state_dict1)
		deprel_model_loc = os.path.join(model_folder, model_state_dict2)
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
			input_ids = []
			attention_mask = []

			word_batch.append(torch.tensor(sent_dict['lemma_ids']).long().to(device))
			word_batch = torch.stack(word_batch).to(device)
			#Collect right inputs for right model
			input_ids.append(torch.tensor(sent_dict['input_ids']).long().to(device))
			attention_mask.append(torch.tensor(sent_dict['attention_mask']).long().to(device))
			input_ids = torch.stack(input_ids).to(device)
			attention_mask = torch.stack(attention_mask).to(device)
			heads = torch.tensor(sent_dict['pair_heads']).long().to(device)
			deprels = torch.tensor(sent_dict['pair_deprel']).long().to(device)
			start_pairs = torch.tensor(sent_dict['start_arc']).long().to(device)
			end_pairs = torch.tensor(sent_dict['end_arc']).long().to(device)

			#Run both models
			#LSTM
			# output_heads = head_classifier.forward(word_batch, start_pairs = start_pairs, end_pairs = end_pairs)
			# output_deps = dep_classifier.forward(word_batch, start_pairs = start_pairs, end_pairs = end_pairs)
			#LM
			output_heads = head_classifier.forward(input_ids = input_ids, attention_mask = attention_mask, start_pairs = start_pairs, end_pairs = end_pairs)
			output_deps = dep_classifier.forward(input_ids = input_ids, attention_mask = attention_mask, start_pairs = start_pairs, end_pairs = end_pairs)

			pred_heads = decode_model(output_heads)
			pred_deps = decode_model(output_deps)

			#Count whether both heads and dependencies match
			for i in range(len(pred_heads)):
				if pred_heads[i] == heads[i] and pred_deps[i] == deprels[i]:
					total_correct += 1
				total_examples += 1
		return ('LAS score: {}').format(total_correct/total_examples)

#TEST CODE FOR TESTING THE EVAL FUNCTION
def test_eval(model_state_dict1, model_state_dict2 = None, model = 'LSTM1', dropout = 0.25, num_layers = 2, hidden_size = 200, test_type = 'UAS', lm = None, base_path = MODEL_FOLDER):
	base_path = os.path.join(base_path, 'UD_English-EWT')

	print('Starting data processing')
	train_lines = preproc_conllu(base_path, filename = 'en_ewt-ud-train.conllu')
	train_sent_collection = sentence_collection(train_lines)
	train_corpus, vocab_dict, label_dict = process_corpus(train_sent_collection, mode = 'train')
	train_corpus = bert_tokenizer(train_corpus)
	train_corpus = form_pairs(train_corpus)

	num_words = len(vocab_dict)
	num_heads = 2
	num_deprel = len(label_dict)

	test_lines = preproc(base_path, filename = 'en_ewt-ud-test.conllu')
	test_sent_collection = sent_collection(test_lines)
	test_corpus, _, _ = process_corpus(test_sent_collection, vocab_dict, label_dict, mode = 'test')
	print('Finished data processing')


	eval_model(test_corpus, num_words,model_state_dict1, model_state_dict2, num_heads, num_deprel, model, dropout, num_layers, hidden_size, test_type, lm)