# file: data_utils.py (RoBERTa 版本)

import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast


def get_tokenizer(model_name="roberta-base"):
    """
    加载 RoBERTa 的官方预训练分词器
    支持远程路径和本地路径
    """
    print(f"正在加载 '{model_name}' 的预训练分词器...")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    print("分词器加载成功。")
    return tokenizer


class FT_Dataset(Dataset):
    """
    适用于 RoBERTa 的数据集类
    """
    def __init__(self, ft_file, batch_size, max_seq_length, tokenizer):
        self.ft_file = ft_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        print(f"正在从 {ft_file} 读取和处理数据...")
        self.samples = self.read_ft_file(ft_file)
        self.num_examples = len(self.samples)
        print(f"找到 {self.num_examples} 个样本。")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item):
        sample = self.samples[item]
        text = sample['text']
        label = sample['label']

        # 使用 RoBERTa 分词器处理文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        output = {
            "input_ids": encoding['input_ids'].squeeze(),
            "attention_mask": encoding['attention_mask'].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return output

    def read_ft_file(self, ft_file):
        if not os.path.exists(ft_file):
            raise FileNotFoundError(f"数据文件未找到: {ft_file}")
        with open(ft_file, 'r', encoding='utf-8') as reader:
            return [json.loads(line.strip()) for line in tqdm(reader, desc=f"正在加载 {os.path.basename(ft_file)}")]
