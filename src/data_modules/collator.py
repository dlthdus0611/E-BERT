#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn.functional as F
import itertools

class Collator(object):
    def __init__(self, args):
        super(Collator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.args = args

    def __call__(self, batch):
        
        lengths_qa = [len(sample['input_ids']) for sample in batch]
        batch_label = []
        batch_input_ids = []
        batch_entity_ids = []
        batch_attention_mask = []
        batch_segment_ids = []

        local_attn_mask = np.zeros((len(batch), max(lengths_qa), max(lengths_qa)))
        for idx, sample in enumerate(batch):
            padding_qa  = [0] * (max(lengths_qa) - len(sample['input_ids']))

            sample['input_ids']  += padding_qa
            sample['entity_ids'] += padding_qa
            sample['attention_mask'] += padding_qa
            sample['token_type_ids'] += padding_qa

            batch_label.append(sample['label'])
            batch_input_ids.append(sample['input_ids'])
            batch_entity_ids.append(sample['entity_ids'])
            batch_attention_mask.append(sample['attention_mask'])
            batch_segment_ids.append(sample['token_type_ids'])
            
            comb = itertools.combinations(sample['entity_idx'], 2)
            for (i,j) in comb:
                local_attn_mask[idx, i, j] = 1
                local_attn_mask[idx, j, i] = 1

        batch_label = torch.LongTensor(batch_label).to(self.device)
        batch_input_ids = torch.LongTensor(batch_input_ids).to(self.device)
        batch_entity_ids = torch.LongTensor(batch_entity_ids).to(self.device)
        batch_attention_mask = torch.LongTensor(batch_attention_mask).to(self.device)
        batch_segment_ids = torch.LongTensor(batch_segment_ids).to(self.device)
        local_attn_mask = torch.LongTensor(local_attn_mask).unsqueeze(1).repeat(1, 12, 1, 1).to(self.device)

        return batch_label, batch_input_ids, batch_entity_ids, batch_attention_mask, batch_segment_ids, local_attn_mask