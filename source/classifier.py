import os
import re
import argparse
import numpy as np
import pandas as pd
import setting
import logging
import logging.config

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from torch import nn
from patterns import regex_pattern_list
from tqdm import tqdm, tqdm_notebook
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.tokenization import *
from torch.optim import Adam

# define logger
# DEBUG < INFO < WARNING < ERROR < CRITICAL
logging.config.fileConfig('logging.conf')
log = logging.getLogger('asa')

# FIXME: all default value should come from setting.py
parser = argparse.ArgumentParser()
parser.add_argument('--train_filename', default=setting.train_filename, help='train filename', type=str)
parser.add_argument('--best_bert_name', default=setting.best_bert_name, help='path to pretrain model', type=str)
parser.add_argument('--model_path', default=setting.model_path, help='output file path', type=str)
parser.add_argument('--vocab_filename', default=setting.vocab_filename, help='vocabulary file path', type=str)
parser.add_argument('--max_len', default=setting.max_len, type=int)
parser.add_argument('--batch_size', default=setting.batch_size, type=int)
parser.add_argument('--num_train_epochs', default=setting.num_train_epochs, type=int)
parser.add_argument('--dist_url', default=setting.dist_url, type=str)
parser.add_argument('--random_state', default=setting.random_state, type=int)
parser.add_argument('--text_col', default=setting.text_col, type=str)
parser.add_argument('--target_col', default=setting.target_col, type=str)
parser.add_argument('--learning_rate', default=setting.learning_rate, type=float)
parser.add_argument('--use_cpu', action='store_false' if setting.use_cpu else 'store_true',
	help='whether to use cpu for loading pretrained model')
parser.add_argument('--multiprocessing-distributed', action='store_true',
	help='Use multi-processing distributed training to launch '
	'N processes per node, which has N GPUs. This is the '
	'fastest way to use PyTorch for either single node or '
	'multi node data parallel training')

class InputExample(object):
	"""A single training/test example for the language model."""

	def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None, labels=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			tokens_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			tokens_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.tokens_a = tokens_a
		self.tokens_b = tokens_b
		self.is_next = is_next  # nextSentence
		self.lm_labels = lm_labels  # masked words for language model
		self.labels = labels


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids, labels=None):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.is_next = is_next
		self.lm_label_ids = lm_label_ids
		self.labels = labels


def convert_classification_example_to_features(example, max_seq_length, tokenizer):
	tokens = example.tokens_a
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1]*len(input_ids)

	pad_idx = tokenizer.vocab['[PAD]']

	if len(input_ids) < max_seq_length:
		n_pad = max_seq_length - len(input_ids)
		input_ids = input_ids + [pad_idx]*n_pad
		input_mask = input_mask + [0]*n_pad
	else:
		input_ids = input_ids[:max_seq_length]
		input_mask = input_mask[:max_seq_length]
	assert len(input_ids) == max_seq_length, 'input_ids has invalid length of {}'.format(len(input_ids))
	assert len(input_mask) == max_seq_length, 'input_mask has invalid length of {}'.format(len(input_mask))

	features = InputFeatures(input_ids=input_ids,
							input_mask=input_mask,
							segment_ids=None,
							lm_label_ids=None,
							is_next=None,
							labels=example.labels)

	return features

class JigsawBertDatasetForClassification(Dataset):
	def __init__(self, 
		df,
		tokenizer,
		seq_len,
		text_col='comment_text',
		target_col='target'):
		self.vocab = tokenizer.vocab
		self.tokenizer = tokenizer
		self.seq_len = seq_len
		self.text_col = text_col
		self.target_col = target_col
		self.sampler_counter = 0
		self.df = df

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, item):
		cur_id = self.sampler_counter

		text = self.df[self.text_col].iloc[item]
		text = self.tokenizer.tokenize(text)

		# combine to one sample
		if self.target_col:
			targets = self.df[self.target_col].iloc[item].tolist()
			example = InputExample(guid=cur_id, tokens_a=text, tokens_b=None, labels=targets)
		else:
			example = InputExample(guid=cur_id, tokens_a=text, tokens_b=None, labels=None)

		# transform sample to features
		features = convert_classification_example_to_features(
			example, 
			self.seq_len, 
			self.tokenizer)
		return features


def preprocess(text):
	text = text.lower()
	for regex in regex_pattern_list:
		pat, substr = regex
		text = re.sub(pat, substr, text)
	return text

def create_features(args, df, tok, has_label=True):
	if has_label:
		target_columns = args.target_col
	else:
		target_columns = None

	features = JigsawBertDatasetForClassification(
		df,
		tok,
		args.max_len,
		text_col=args.text_col,
		target_col=args.target_col)

	all_input_ids = []
	all_input_mask = []
	all_labels = []
	for f in tqdm(features, desc='generate_torch_tensor'):
		all_input_ids.append(f.input_ids)
		all_input_mask.append(f.input_mask)
		all_labels.append(f.labels)
		if len(f.input_mask) != 128:
			log.debug('input_mask: {}'.format(f.input_mask))

	all_input_ids = torch.LongTensor(all_input_ids)
	all_input_mask = torch.LongTensor(all_input_mask)
	if has_label:
		all_labels = torch.LongTensor(all_labels)
		log.debug('all_input_ids: {}'.format(len(all_input_ids)))
		log.debug('all_input_mask: {}'.format(len(all_input_mask)))
		log.debug('all_labels: {}'.format(len(all_labels)))
		return [all_input_ids, all_input_mask, all_labels]
	else:
		all_labels = None
		log.debug('all_input_ids: {}'.format(len(all_input_ids)))
		log.debug('all_input_mask: {}'.format(len(all_input_mask)))
		return [all_input_ids, all_input_mask]

def create_dataloader(torch_features, args, use_sampler=True):
	tensor_data = TensorDataset(*torch_features)
	log.debug('TensorDataset ready')

	log.debug('create_dataloader -> use_sampler={}'.format(use_sampler))
	if use_sampler:
		sampler = RandomSampler(tensor_data)
		log.debug('sampler ready')

		dataloader = DataLoader(tensor_data, sampler=sampler, batch_size=args.batch_size)
		log.debug('DataLoader ready')
		return dataloader, sampler
	else:
		dataloader = DataLoader(tensor_data, sampler=None, batch_size=args.batch_size)
		log.debug('DataLoader ready')
		return dataloader


def main():
	args = parser.parse_args()

	# load dataset
	df = pd.read_csv('/data/jigsaw/train_sample.csv')
	df = df.sample(frac=0.2, random_state=args.random_state)
	log.debug('dataset columns: {}'.format(df.columns))

	# FIXME: you may not need to make it int
	tqdm.pandas()
	df.loc[:, args.text_col] = df[args.text_col].progress_apply(preprocess)
	df.loc[:, args.target_col] = df[args.target_col].progress_apply(lambda x: 0 if x < 0.5 else 1)
	#log.debug('dataset target {}: {}'.format(args.target_col, df[args.target_col]))

	# split dataframe
	n_train = int(df.shape[0] * 0.8)
	train_df, valid_df = df[:n_train], df[n_train:]
	log.debug('train dataframe: {}'.format(train_df.shape))
	log.debug('valid dataframe: {}'.format(valid_df.shape))

	# load tokenizer
	tok = BertTokenizer.from_pretrained(args.best_bert_name, do_lower_case=True)
	log.debug('tokenizer: {}'.format(tok))

	# make dataloader
	train_feature = create_features(args, train_df, tok)
	valid_feature = create_features(args, valid_df, tok)

	#train_dataloader
	train_dataloader, train_sampler = create_dataloader(train_feature, args)
	valid_dataloader, _ = create_dataloader(valid_feature, args)
	log.debug('train_dataloader: {}'.format(train_dataloader))
	log.debug('valid_dataloader: {}'.format(valid_dataloader))

	# define model
	model = BertForSequenceClassification.from_pretrained(args.best_bert_name, num_labels=2)
	model.cuda()
	log.debug('model loaded: {}'.format(model))

	# define optimizer
	optimizer = Adam(model.parameters(), lr=args.learning_rate)

	# train model
	t = tqdm_notebook(range(args.num_train_epochs), desc='Training')
	best_result = 0.0
	for epoch in t:
		train_classifier_model(model, train_dataloader, optimizer)
		labels, predictions = validate_classifier_model(model, valid_dataloader)

		'''
		# calculate and display evaluation result
		df = display_validation_result(labels, predictions)
		log.info('validation result at {}th epoch: \n{}'.format(epoch, df))
		current_result = df[['precision','recall']].values.flatten().mean()

		# save classifier model
		is_best = True if current_result > best_result else False
		best_result = max(current_result, best_result)
		log.debug('best_result: {}'.format(best_result))

		save_classifier_checkpoint({
			'state_dict': model.state_dict(),
		}, args, is_best, filename=model_name_with_fullpath, rand=randidx)
		'''
		break



if __name__ == '__main__':
	main()
