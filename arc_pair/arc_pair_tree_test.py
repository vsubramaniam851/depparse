import os
import sys
import random
import numpy as np 
import csv
import argparse

import torch
import torch.nn as nn
import torch.functional as F 
import torch.cuda as cuda 

import transformers

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def is_well_formed(self, edges):
	if len(edges) == 0:
		return False  
	seen1 = []
	seen2 = [] 
	for pair in edges:
		start, end = pair
		if end in seen1:
			return False
	return True

def get_cmd_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = '/storage/vsub851/depparse',
		help = 'Base path with the data and models')
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'model', default = 'LM1', 
		help = 'The model to run either LSTM1, LSTM2, LM1, LM2 for each corresponding model')
	ap.add_argument('-hm', '--headmodel', action = 'store', type = str, dest = 'head_model', default = 'headlmclassifier1.pt',
		help = 'Pytorch model that predicts the heads that can be saved to after training or loaded in for evaluation')
	ap.add_argument('-d', '--deprelmodel', action = 'store', type = str, dest = 'deprel_model', default = 'deprelclassifier1.pt',
		help = 'Pytorch model that predicts the dependency labels that can be saved to after training or loaded in for evaluation')
	ap.add_argument('-s', '--savecsv', action = 'store', type = bool, dest = 'save_csv', default = False, 
		help = 'Save processed CoNLLU files to extract more easily later.')

	ap.add_argument('--batch_size', type=int, default=1, action = 'store', dest = 'batch_size',
		help = 'Change from 1 only if you use lemma padding')
	ap.add_argument('--num_layers', type=int, default=3, action = 'store', dest = 'num_layers')
	ap.add_argument('--hidden_size', type=int, default=400, action = 'store', dest = 'hidden_size')
	ap.add_argument('--dropout', type=float, default=0.33, action = 'store', dest=  'dropout')

	return ap.parse_args()

def main():
	args = get_cmd_arguments()
	data_loc = os.path.join(base_path, 'UD_English-EWT')
	#Load training file
	train_lines = preproc_conllu(data_loc, filename = 'en_ewt-ud-train.conllu', save_csv = args.save_csv)
	#Load testing file, save to CSV unless already saved.
	test_lines = preproc_conllu(data_loc, filename = 'en_ewt-ud-test.conllu', save_csv = args.save_csv)
	#DATA PROCESSING PIPELINE FOR BOTH TRAINING AND TESTING FILES
	train_sent_collection = sentence_collection(train_lines)
	test_sent_collection = sentence_collection(test_lines)

	train_corpus, vocab_dict, label_dict = process_corpus(train_sent_collection, mode = 'train')
	test_corpus, _, _= process_corpus(test_sent_collection, vocab_dict = vocab_dict, label_dict = label_dict)

	#Tokenize using BERT
	train_corpus = bert_tokenizer(train_corpus)
	test_corpus = bert_tokenizer(test_corpus)

	#Get pairs for the train and test corpus
	train_corpus = form_pairs(train_corpus)
	test_corpus = form_pairs(test_corpus)
	print('Data processing is finished')

	if args.model == 'LM1' or args.model == 'LM2':
		lm = lm_pretrained
	else:
		lm = None

	num_words = len(vocab_dict)
	num_heads = 2
	num_deprel = 0

	new_base_path = os.path.join(base_path, 'arc_pair')
	if model == 'LSTM1':
		classifier = LSTMClassifier1(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
	elif model == 'LSTM2':
		classifier = LSTMClassifier2(num_words, num_heads, num_layers, hidden_size, dropout = dropout)
	elif model == 'LM1':
		classifier = LMClassifier1(num_labels = num_heads, hidden_size = hidden_size, lm = lm, dropout = dropout)
	else:
		classifier = LMClassifier2(num_labels = num_heads, hidden_size = hidden_size, lm = lm, num_layers = num_layers, dropout = dropout)

	model_folder = os.path.join(new_base_path, 'checkpoints')
	model_loc = os.path.join(model_folder, model_state_dict1)
	classifier.load_state_dict(torch.load(model_loc))
	classifier = classifier.to(device)

	label_type = 'pair_heads'

	head_dep_pair_sent = []
	all_spans = []
	for i in range(len(test_corpus)):
		sent_dict = test_corpus[i]
		word_batch = []
		sent_len = len(sent_dict['lemma_ids'])
		sent = sent_dict['sent']
		# word_batch.append(torch.tensor(sent_dict['lemma_ids']).long().to(device))
		# word_batch = torch.stack(word_batch).to(device)
		input_ids = []
		attention_mask = []
		#Collect right inputs for right model
		input_ids.append(torch.tensor(sent_dict['input_ids']).long().to(device))
		attention_mask.append(torch.tensor(sent_dict['attention_mask']).long().to(device))
		input_ids = torch.stack(input_ids).to(device)
		attention_mask = torch.stack(attention_mask).to(device)
		start_pairs = torch.tensor(sent_dict['start_arc']).long().to(device)
		end_pairs = torch.tensor(sent_dict['end_arc']).long().to(device)

		#Run and decode classifier
		#LSTM
		# outputs = classifier.forward(word_batch, start_pairs, end_pairs)
		#LM
		outputs = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, start_pairs = start_pairs, end_pairs = end_pairs)
		#Decode model to get predictions of which pairs are head-dependent
		preds = decode_model(outputs)

		spans = []
		head_dep_pairs = []
		for i in range(len(predicted_heads)):
			if predicted_heads[i] == 1:
				start, end = pairs[i]
				spans.append((start, end))
				head_dep_pairs.append((sent[start], sent[end]))
		head_dep_pair_sent.append(head_dep_pairs)
		all_spans.append(spans)

	print(head_dep_pair_sent[0])

if __name__ == '__main__':
	main()