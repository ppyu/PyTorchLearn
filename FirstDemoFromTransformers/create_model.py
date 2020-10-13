# -*- coding: utf-8 -*-
"""
@File   : create_model.py
@Author : Pengy
@Date   : 2020/10/13
@Description : Input your description here ... 
"""
import os
import torch
from FirstDemoFromTransformers.SimpleSquadModel import SimpleSquadModel
from transformers import BertConfig


def create_model(args, device, config_file='', weights_file=''):
    ''' create squad model from args '''
    ModelClass = None
    if args.squad_model == 'bert_base':
        print('creating bert base model')
        ModelClass = SimpleSquadModel
    if args.squad_model == 'bert_linear':
        print('creating bert linear model')
        # ModelClass = SquadLinearModel
    if args.squad_model == 'bert_deep':
        print('creating bert deep model')
        # ModelClass = SquadDeepModel
    if args.squad_model == 'bert_qanet':
        print('creating bert qanet model')
        # ModelClass = SquadModelQANet

    if config_file == '' and weights_file == '':
        print('creating an untrained model')
        return ModelClass.from_pretrained(args.bert_model)
    else:
        print('loading a trained model')
        config = BertConfig(config_file)
        model = ModelClass(config)
        model.load_state_dict(torch.load(weights_file, map_location=device))
        return model
