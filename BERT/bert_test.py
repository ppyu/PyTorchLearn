# -*- coding: utf-8 -*-
"""
@File   : bert_test.py
@Author : Pengy
@Date   : 2020/10/12
@Description : Input your description here ... 
"""
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# 可选：如果您想了解发生的信息，请按以下步骤logger
import logging

logging.basicConfig(level=logging.INFO)

# 加载预训练的模型标记器（词汇表）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 标记输入
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# 用“BertForMaskedLM”掩盖我们试图预测的标记`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet',
                          '##eer', '[SEP]']

# 将标记转换为词汇索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# 定义与第一句和第二句相关的句子A和B索引（见论文）
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 将输入转换为PyTorch张量
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# 加载预训练模型（权重）
model = BertModel.from_pretrained('bert-base-uncased')

# 将模型设置为评估模式
# 在评估期间有可再现的结果这是很重要的！
model.eval()

# 如果你有GPU，把所有东西都放在cuda上
if torch.cuda.is_available():
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

# 预测每个层的隐藏状态特征
with torch.no_grad():
    # 有关输入的详细信息，请参见models文档字符串
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # Transformer模型总是输出元组。
    # 有关所有输出的详细信息，请参见模型文档字符串。在我们的例子中，第一个元素是Bert模型最后一层的隐藏状态
    encoded_layers = outputs[0]
    print(encoded_layers.shape)
# 我们已将输入序列编码为形状（批量大小、序列长度、模型隐藏维度）的FloatTensor
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)

