import sys
import os
import numpy as np 
import csv

import torch
import torch.nn as nn
import torch.functional as F 

import transformers

#Use BERT tokenizer for tokenizing sentences
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

#Reset base path to whatever directory has the data
base_path = '/storage/vsub851/864_final_project/UD_English-EWT'

def preproc_conllu(base_path, filename, save_csv = False):
	'''Open the conllu file and filter out all comment lines in the conllu.
	If save_csv = True, write the filtered lines to new conllu in data_conllu'''
	file_path = os.path.join(base_path, filename)
	f = open(file_path, 'r')
	lines = []
	for line in f.readlines():
		if line[0] != '#':
			#Strip all new lines from the file
			line = line.strip('\n') 
			#CoNLL-U files are tab delimited
			lines.append(line.split('\t')) 
	print('Finished processing Conllu')
	#Save to new CoNLL-U without comment lines so that we only have the text
	if save_csv:
		print('Saving to new Conllu')
		save_path = os.path.join(base_path, 'data_conllu', filename)
		with open(save_path, mode = 'w') as write_conllu:
			writer = csv.writer(write_conllu, delimiter = '\t')

			for l in lines:
				writer.writerow(l)
	return lines

def sentence_collection(lines):
	'''Collect the lines of the files into sentence collections for input to the tokenizer'''
	sentences = []
	new_sent = []
	for l in lines:
		#CoNLL-U lines describe a word and the first entry in the file describes the index of that word in the sentence. If the first entry is 1, 
		# this indicates a new sentence
		if l[0] == '1':
			sentences.append(new_sent)
			if [] in sentences:
				#Remove any empty lines that indicate a break between sentences
				sentences.remove([])
			new_sent = [l]
		else:
			if l != ['']:
				new_sent.append(l)
	return sentences 

def process_corpus(corpus, mode = None, vocab_dict = None, label_dict = None):
	'''Take in the sentences which are in conllu format and collect the words, lemmas, heads, and dependencies.
	The word is the first index and will be used for the tokenizer and language model pipeline. The lemmas will be used 
	for the LSTM sentence encoding.
	'''
	if not vocab_dict:
		#Initialize vocab dict if we are training the model
		vocab_dict = {'UNK': 0}
	if not label_dict:
		label_dict = {}
	sent_parses = []
	for sent in corpus:
		deprel_ids = []
		lemma_ids = []
		heads = []
		words = []
		length = len(sent)
		for word in sent:
			#Skip all blank lines
			if len(word) <= 1:
				continue
			words.append(word[1])
			lemma = word[2]
			head = word[6]
			deprel = word[7]
			if head == '_':
				#If a head is _ that means it is blank or unknown. Just append the length of the sentence for this instance
				head = str(length)
			heads.append(int(head))
			if lemma in vocab_dict:
				lemma_ids.append(vocab_dict[lemma])
			elif mode == 'train':
				#Create corpus if we are going to train a new model
				lemma_ids.append(len(vocab_dict))
				vocab_dict[lemma] = len(vocab_dict)
			else:
				#If we are processing the test corpus, any word that is not in the corpus is marked as UNK
				lemma_ids.append(vocab_dict['UNK'])
			if deprel in label_dict:
				deprel_ids.append(label_dict[deprel])
			else:
				deprel_ids.append(len(label_dict))
				label_dict[deprel] = len(label_dict)
		#Create sentence from joining the words since we may use a language model to get input ids which needs the sentence
		sentence = ' '.join(words)
		#Add a dictionary for each sentence in the corpus. This will represent our data corpus for all sentences
		sent_parses.append({'sent': words, 'lemma_ids': lemma_ids, 'deprel_ids': deprel_ids, 'heads': heads, 'joined': sentence})
	return sent_parses, vocab_dict, label_dict 

def num_heads(sent_parses):
	'''Find the longest sentence since this is the number of labels for when we predict heads'''
	max_length = 0
	for i in range(len(sent_parses)):
		if len(sent_parses[i]['heads']) > max_length:
			max_length = len(sent_parses[i]['heads'])
	return max_length

def lemma_padding(sent_parses, vocab_dict):
	'''Pad lemmas in case we want to change the batch size'''
	num_words = len(vocab_dict)
	for i in range(len(sent_parses)):
		#Add num word to pad the ID
		sent_parses[i]['lemma_ids'].append(num_words)
	return sent_parses 

def bert_tokenizer(sent_parses):
	'''Take in the parsed sentence and tokenize using BERT.'''
	sents = []
	new_sent_parses = []
	for dicts in sent_parses:
		sent = dicts['sent']
		#Use BERT base uncased to encode the sentence by accessing input IDs and attention mask. Do not set a max length of truncate since this will not change
		#The length of the sentence
		encoding = tokenizer.encode_plus(sent, return_attention_mask = True)
		#Create new corpus with input ids without sentence since it is unnecessary.
		new_sent_parses.append({'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'lemma_ids': dicts['lemma_ids'], 'deprel_ids': dicts['deprel_ids'], 'heads': dicts['heads']})
	return new_sent_parses

#TESTING THE DATA LOADING CODE 
def test_data_loading(base_path, filename):
	#CoNLL-U preprocessing
	lines = preproc_conllu(base_path, filename = 'small.conllu')

	#Collect sentences from CoNLLU
	sent_collection = sentence_collection(lines)
	print(sent_collection)

	#Process the corpus
	sent_parses, vocab_dict, label_dict = process_corpus(sent_collection, mode = 'train')

	#Check head padding
	max_length = num_heads(sent_parses)
	print('THIS IS MAX LENGTH', max_length)

	#Check lemma padding
	padded_sent_parses = lemma_padding(sent_parses, vocab_dict)
	for i in range(len(sent_parses)):
		print(len(padded_sent_parses[i]['lemmas']))

	#Check BERT tokenizer
	new_sent_parses = bert_tokenizer(sent_parses)
	for i in range(len(new_sent_parses)):
		sent_dict = new_sent_parses[i]
		print('INPUT IDS:', len(sent_dict['input_ids']))
		print('LEMMA IDS:', len(sent_dict['lemma_ids']))