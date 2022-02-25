#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals)

import torch
import os
import numpy as np

# generate config
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]

train_file = root_path + '/data/train_clean.tsv'
dev_file = root_path + '/data/dev_clean.tsv'
test_file = root_path + '/data/test_clean.csv'
stopWords_file = root_path + '/data/stopwords.txt'
log_dir = root_path + '/logs/'

# generate dl config
embedding = 'random'
embedding_pretrained = torch.tensor(
                       np.load(root_path + '/data/' + embedding)["embeddings"].astype('float32')) \
                       if embedding != 'random' else None

is_cuda = False
device = torch.device('cuda') if is_cuda else torch.device('cpu')
class_list = [
    x.strip() for x in open(root_path + '/data/class.txt', encoding='utf-8').readlines()
]   
num_classes = len(class_list)

num_epochs = 30   
batch_size = 32   
pad_size = 400   
learning_rate = 2e-5   
dropout = 1.0   
require_improvement = 10000   
n_vocab = 50000   
embed = 300   
hidden_size = 512   
num_layers = 1   
eps = 1e-8
max_length = 400
dim_model = 300
hidden = 1024
last_hidden = 512
num_head = 5
num_encoder = 2

