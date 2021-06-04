import sys
import os
import numpy as np 
import csv

import torch
import torch.nn as nn
import torch.functional as F 

import transformers

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

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
		vocab_dict = {'UNK': 0, 'ROOT': 1}
	if not label_dict:
		label_dict = {'NONE': 0}
	sent_parses = []
	for sent in corpus:
		deprel_ids = [0]
		lemma_ids = [vocab_dict['ROOT']]
		heads = [0]
		words = ['ROOT']
		length = len(sent)+1
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

def bert_tokenizer(sent_parses):
	'''Take in the parsed sentence and tokenize using BERT.'''
	sents = []
	new_sent_parses = []
	for dicts in sent_parses:
		sent = dicts['sent'][1:]
		#Use BERT base uncased to encode the sentence by accessing input IDs and attention mask. Do not set a max length of truncate since this will not change
		#The length of the sentence
		encoding = tokenizer.encode_plus(sent, return_attention_mask = True)
		#Create new corpus with input ids without sentence since it is unnecessary.
		new_sent_parses.append({'sent': dicts['sent'], 'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'lemma_ids': dicts['lemma_ids'], 'deprel_ids': dicts['deprel_ids'], 'heads': dicts['heads']})
	return new_sent_parses

def form_pairs(sent_parses):
	new_sent_parses = []
	for sent in sent_parses:
		pairs = []
		head_val = []
		deprel_val = []
		words = sent['sent']
		heads = sent['heads']
		deprel_ids = sent['deprel_ids']
		for i in range(len(words)):
			for j in range(len(words)):
				pairs.append((i,j))
				if j == heads[i]:
					head_val.append(1)
					deprel_val.append(deprel_ids[i])
				else:
					head_val.append(0)
					deprel_val.append(0)
		start_arc = []
		end_arc = []
		for start, end in pairs:
			start_arc.append(start)
			end_arc.append(end)
		new_sent_parses.append({'pairs': pairs, 'pair_heads': head_val, 'start_arc': start_arc, 'end_arc': end_arc, 'pair_deprel':deprel_val, 'sent': sent['sent'], 'input_ids': sent['input_ids'], 'attention_mask': sent['attention_mask'], 'lemma_ids': sent['lemma_ids'], 'deprel_ids': sent['deprel_ids'], 'heads': sent['heads']})
	return new_sent_parses

def data_load_test(base_path, filename, save_csv = False, mode = 'train'):
	lines = preproc_conllu(base_path, filename, save_csv)
	print('Lines in ConLL-U', lines)

	sentences = sentence_collection(lines)
	print('Sentences', sentences)

	corpus, vocab_dict, label_dict = process_corpus(sentences, mode = 'train')
	print('First example from corpus', corpus[0])
	print(len(vocab_dict))

	bert_corpus = bert_tokenizer(corpus)
	print('BERT tokenized corpus', corpus[0])

	train_corpus = form_pairs(bert_corpus)
	print('Full train corpus', train_corpus[0])

# data_load_test('/storage/vsub851/depparse/UD_English-EWT', 'en_ewt-ud-train.conllu')