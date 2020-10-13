# -*- coding: utf-8 -*-
"""
@File   : train.py
@Author : Pengy
@Date   : 2020/10/13
@Description : Input your description here ... 
"""
import torch
from torch import nn
import numpy as np
import random
import os
from FirstDemoFromTransformers.args import get_setup_args, get_train_args, check_args
from transformers import BertTokenizer, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV2Processor


def main(args):
    # 检查参数合法性
    check_args(args)
    # 检查GPU环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # 创建模型输出路径
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_optimization_steps = None
    # 使用transformers库提供的针对squad数据集的预处理器处理数据
    squad_data_processor = SquadV2Processor()
    examples = squad_data_processor.get_train_examples(args.data_dir, filename=args.train_file)
    features, train_dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
    )
    train_sampler = RandomSampler(train_dataset)



if __name__ == '__main__':
    main()
