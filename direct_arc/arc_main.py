import os
import sys
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.functional as F  

from arc_train import *
from arc_eval import *
from data_loader import *

import transformers

#Set device to cuda on all files. Run on GPU if possible
if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

base_path = '/storage/vsub851/864_final_project'

#Use BERT base uncased to keep sentence length the same
lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def get_cmd_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'model', default = 'LSTM1', 
		help = 'The model to run either LSTM1, LSTM2, LM1, LM2 for each corresponding model')
	ap.add_argument('-e', '--eval', action = 'store', type = str, dest = 'eval_type', default = 'UAS', 
		help = 'The type of evaluation, either UAS or LAS')
	ap.add_argument('-t', '--train', action = 'store', type = bool, dest = 'train_model', default = False, 
		help = 'Set to true if we want to train the model')
	ap.add_argument('-hm', '--headmodel', action = 'store', type = str, dest = 'head_model', default = 'headclassifier1.pt',
		help = 'Pytorch model that predicts the heads that can be saved to after training or loaded in for evaluation')
	ap.add_argument('-d', '--deprelmodel', action = 'store', type = str, dest = 'deprel_model', default = 'deprelclassifier1.pt',
		help = 'Pytorch model that predicts the dependency labels that can be saved to after training or loaded in for evaluation')

	#Model Hyperparameters
	ap.add_argument('--batch_size', type=int, default=1, action = 'store', dest = 'batch_size',
		help = 'Change from 1 only if you use lemma padding')
	ap.add_argument('--num_layers', type=int, default=2, action = 'store', dest = 'num_layers')
	ap.add_argument('--hidden_size', type=int, default=200, action = 'store', dest = 'hidden_size')
	ap.add_argument('--lr', type=float, default=0.00005, action = 'store', dest = 'lr')
	ap.add_argument('--epochs', type=int, default=20, action = 'store', dest = 'num_epochs')
	ap.add_argument('--dropout', type=float, default=0.33, action = 'store', dest=  'dropout')

	return ap.parse_args()

def main():
	args = get_cmd_arguments()
	print('Start data processing')
	#Get path to data directory
	data_loc = os.path.join(base_path, 'UD_English-EWT')
	#Load training file
	train_lines = preproc_conllu(data_loc, filename = 'en_ewt-ud-train.conllu')
	#Load testing file, save to CSV unless already saved.
	test_lines = preproc_conllu(data_loc, filename = 'en_ewt-ud-test.conllu', save_csv = True)

	#DATA PROCESSING PIPELINE FOR BOTH TRAINING AND TESTING FILES
	train_sent_collection = sentence_collection(train_lines)
	test_sent_collection = sentence_collection(test_lines)

	train_corpus, vocab_dict, label_dict = process_corpus(train_sent_collection, mode = 'train')
	test_corpus, _, _= process_corpus(test_sent_collection, vocab_dict = vocab_dict, label_dict = label_dict)

	#Pad heads based on which dataset (training or testing) has the longest sentence
	if num_heads(train_corpus) >= num_heads(test_corpus):
		head_labels = num_heads(train_corpus)
	else:
		head_labels = num_heads(test_corpus)


	#Tokenize using BERT
	train_corpus = bert_tokenizer(train_corpus)
	test_corpus = bert_tokenizer(test_corpus)
	print('Data processing is finished')

	#If the model is a Language model, set it to the lm_pretrained
	if args.model == 'LM1' or args.model == 'LM2':
		lm = lm_pretrained
	else:
		lm = None

	num_words = len(vocab_dict)
	num_labels = head_labels + 1
	num_labels1 = 0
	if args.eval_type == 'LAS':
		#Labeled attachment score needs the dependency model to be trained as well.
		num_labels1 = len(label_dict)
	if args.train_model:
		#Train the model
		if args.eval_type == 'UAS':
			arc_train(train_corpus, train_type = 'heads', num_words = num_words, num_labels = num_labels, modelname = args.head_model, hidden_size = args.hidden_size, lr = args.lr, dropout = args.dropout, num_epochs = args.num_epochs, 
				num_layers = args.num_layers, batch_size = args.batch_size, model = args.model, lm = lm)
		else:
			arc_train(train_corpus, train_type = 'heads', num_words = num_words, num_labels = num_labels, modelname = args.head_model, hidden_size = args.hidden_size, lr = args.lr, dropout = args.dropout, num_epochs = args.num_epochs, 
				num_layers = args.num_layers, batch_size = args.batch_size, model = args.model, lm = lm)
			arc_train(train_corpus, train_type = 'deprel', num_words = num_words, num_labels = num_labels1, modelname = args.deprel_model, hidden_size = args.hidden_size, lr = args.lr, dropout = args.dropout, num_epochs = args.num_epochs, 
				num_layers = args.num_layers, batch_size = args.batch_size, model = args.model, lm = lm)
	else:
		#If model is already trained, evaluate the model
		if args.eval_type == 'UAS':
			print(eval_model(test_corpus, num_words = num_words, model_state_dict1 = args.head_model, model_state_dict2 = args.deprel_model, num_heads = num_labels, num_deprel = num_labels1, model = args.model, dropout = args.dropout, num_layers = args.num_layers, hidden_size = args.hidden_size, test_type = args.eval_type, lm = lm))
		else:
			print(eval_model(test_corpus, num_words = num_words, model_state_dict1 = args.head_model, model_state_dict2 = args.deprel_model, num_heads = num_labels, num_deprel = num_labels1, model = args.model, 
				dropout = args.dropout, num_layers = args.num_layers, hidden_size = args.hidden_size, test_type = args.eval_type))

if __name__ == '__main__':
    main()