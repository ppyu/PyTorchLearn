# -*- coding: utf-8 -*-
"""
@File   : bert_return_test.py
@Author : Pengy
@Date   : 2020/10/13
@Description : 测试bert的返回值
"""
from transformers import BertTokenizer, BertModel
import torch
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs[1].size())
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.size())
print(PYTORCH_PRETRAINED_BERT_CACHE)
