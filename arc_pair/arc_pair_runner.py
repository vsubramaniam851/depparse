import os
import sys
import numpy as np 
import nltk
from pywsd.utils import lemmatize_sentence
import argparse

import torch
import torch.nn as nn
import torch.functional as F 
from torch import cuda 

import transformers

from data_loader import *
from arc_pair_model import *
from arc_pair_eval import *

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def get_cmd_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = '/storage/vsub851/depparse',
		help = 'Base path with the data and models')
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'model', default = 'LM1', 
		help = 'The model to run either LSTM1, LSTM2, LM1, LM2 for each corresponding model')
	ap.add_argument('-hm', '--headmodel', action = 'store', type = str, dest = 'head_model', default = 'head_lm1_20.pt',
		help = 'Pytorch model that predicts the heads that can be saved to after training or loaded in for evaluation')
	ap.add_argument('-d', '--deprelmodel', action = 'store', type = str, dest = 'deprel_model', default = 'deprelclassifier1.pt',
		help = 'Pytorch model that predicts the dependency labels that can be saved to after training or loaded in for evaluation')
	ap.add_argument('-r', '--runtype', action = 'store', type = str, dest = 'run_type', default = 'heads',
		help = 'Run just head model or both head and deprel model')

	ap.add_argument('--batch_size', type=int, default=1, action = 'store', dest = 'batch_size',
		help = 'Change from 1 only if you use lemma padding')
	ap.add_argument('--num_layers', type=int, default=3, action = 'store', dest = 'num_layers')
	ap.add_argument('--hidden_size', type=int, default=400, action = 'store', dest = 'hidden_size')
	ap.add_argument('--dropout', type=float, default=0.33, action = 'store', dest=  'dropout')

	return ap.parse_args()

def sent_process(base_path, sent, model):
	sent_lemmas = lemmatize_sentence(sent)
	data_loc = os.path.join(base_path, 'UD_English-EWT')
	#Load training file
	train_lines = preproc_conllu(data_loc, filename = 'en_ewt-ud-train.conllu')

	train_sent_collection = sentence_collection(train_lines)
	_, vocab_dict, label_dict = process_corpus(train_sent_collection, mode = 'train')
	if model == 'LSTM1' or model == 'LSTM2':
		lemma_ids = []
		for w in sent_lemmas:
			if w in vocab_dict:
				lemma_ids.append(vocab_dict[w])
			else:
				lemma_ids.append(vocab_dict['UNK'])
		return (lemma_ids, len(vocab_dict))
	else:
		sent = sent.split()
		sent_encoding = tokenizer.encode_plus(sent)
		return (sent_encoding['input_ids'], sent_encoding['attention_mask'], len(vocab_dict))

def is_well_formed(self, spans):
	if len(spans) == 0:
		return False   
	for s1 in spans: 
		for s2 in spans:
			if s1[0] < s2[0] and s2[0] < s1[1] and s1[1] < s2[1]:
				return False
	return True

def runner(sent):
	args = get_cmd_arguments()
	base_path = args.base_path
	model = args.model

	processed_sent = sent_process(base_path, sent, model)

	if args.model == 'LM1' or args.model == 'LM2':
		lm = lm_pretrained
	else:
		lm = None

	num_layers = args.num_layers
	hidden_size = args.hidden_size
	dropout = args.dropout
	num_words = processed_sent[-1]
	num_heads = 2

	sent = sent.split()
	pairs = []
	start_pairs = []
	end_pairs = []
	for i in range(len(sent)):
		for j in range(len(sent)):
			pairs.append((i, j))
	for i,j in pairs:
		start_pairs.append(i)
		end_pairs.append(j)
	start_pairs = torch.tensor(start_pairs).to(device)
	end_pairs = torch.tensor(end_pairs).to(device)

	if args.run_type == 'heads':
		if model == 'LSTM1':
			classifier = LSTMClassifier1(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
		elif model == 'LSTM2':
			classifier = LSTMClassifier2(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
		elif model == 'LM1':
			classifier = LMClassifier1(num_labels = num_heads, hidden_size = hidden_size, lm = lm, dropout = dropout)
		else:
			classifier = LMClassifier2(num_labels = num_heads, hidden_size = hidden_size, lm = lm, num_layers = num_layers, dropout = dropout)

		model_folder = os.path.join(base_path, 'arc_pair', 'checkpoints')
		model_loc = os.path.join(model_folder, args.head_model)
		classifier.load_state_dict(torch.load(model_loc))
		classifier = classifier.to(device)

		if model == 'LM1' or 'LM2':
			input_ids, attention_mask, _ = processed_sent
			input_ids = torch.stack([torch.tensor(input_ids)]).to(device)
			attention_mask = torch.stack([torch.tensor(attention_mask)]).to(device)
			model_output = classifier(start_pairs, end_pairs, input_ids, attention_mask)
		else:
			lemma_ids = processed_sent[0]
			lemma_ids = torch.tensor(lemma_ids)
			model_output = classifier(processed_sent[0])

		predicted_heads = decode_model(model_output)

		spans = []
		head_dep_pairs = []
		for i in range(len(predicted_heads)):
			if predicted_heads[i] == 1:
				start, end = pairs[i]
				spans.append((start, end))
				head_dep_pairs.append((sent[start], sent[end]))
		print(is_well_formed(spans))

if __name__ == '__main__':
	sent = input('Enter a sentence to parse: ')
	runner(sent)