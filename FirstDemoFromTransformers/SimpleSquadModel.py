# -*- coding: utf-8 -*-
"""
@File   : SimpleSquadModel.py
@Author : Pengy
@Date   : 2020/10/13
@Description : Input your description here ... 
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel


class SimpleSquadModel(BertPreTrainedModel):
    """Bert model for Question Answering (span extration) on Squad2.0 Dataset
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits.
    """

    def __init__(self, config):
        super(SimpleSquadModel, self).__init__(config)
        self.bert = BertModel(config)
        # self.bert = BertModel.from_pretrained(model_name_or_path, config=config)
        self.qa_output = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        """

        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param start_position:
        :param end_position:
        :return:
        last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) – Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        """
        bert_sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.qa_output(bert_sequence_output)
        # split()根据长度在dim维上去拆分tensor
        start_logits, end_logits = logits.split(1, dim=-1)
        # squeeze()在指定维度压缩一个"1"
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            # torch.clamp(input, min, max, out=None) → Tensor
            # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            # 定义损失函数
            loss_func = CrossEntropyLoss(ignored_index=ignored_index)   #该损失函数结合了nn.LogSoftmax()和nn.NLLLoss()两个函数。它在做分类（具体几类）训练的时候是非常有用的。
            start_loss = loss_func(start_logits, start_positions)
            end_loss = loss_func(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
