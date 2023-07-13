import os
import re
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# augmentation for QA sequences
class Augment:
    def __init__(self, args, shuffle=True, ratio=0.3, qa=True, etc=True, mode='train'):

        self.args    = args
        self.pattern = '{S'
        self.shuffle = shuffle
        self.ratio   = args.ratio
        self.qa      = qa
        self.etc     = etc
        self.mode    = mode

    def _get_pair(self, x):
        
        idx = [m.start(0) for m in re.finditer(self.pattern, x)]
        idx.append(len(x))
        
        x_list = []
        for i in range(len(idx)-1):
            start = idx[i]
            end   = idx[i+1]
            x_list.append(x[start:end])
            
        return x_list
    
    def _augment(self, x, y, tokenizer):

        x = self._get_pair(x)
        keep_idx = range(len(x))
        
        if self.mode == 'train':
            # shuffle sequences
            if self.shuffle:
                x_idx = [i for i in range(len(x))]
                random.shuffle(x_idx)
                x = [x[i] for i in x_idx]
                y = [y[i] for i in x_idx]
            
            # drop sequences
            keep_num = int(len(x) * (1 - self.ratio))
            keep_idx = sorted(random.sample(range(len(x)), keep_num))
            x = [x[i] for i in keep_idx]
            y = [y[i] for i in keep_idx]
        else:
            pass

        tokens = []
        entities = []
        for xx,yy in zip(x,y):
            tmp = tokenizer.tokenize(xx.replace('{S','').replace('E}',''))
            tokens.extend(tmp)
            tokens.append('[SEP]')

            entities.extend(yy)
            entities.append('O')
        
        return tokens, entities
        
    def forward(self, qa, etc:list, tokenizer):

        qa, etc = self._augment(qa, etc, tokenizer)
        
        return qa, etc

class DiagDataset(Dataset):
    def __init__(self, args, data, tokenizer, transforms=None):

        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.transforms = transforms

        data['questions'] = data['questions'].astype('str')
        data['answers'] = data['answers'].astype('str')

        self.data = data

        self.id_encoder = LabelEncoder()
        data['uni_id'] = self.id_encoder.fit_transform(data['uni_id'])

        # ---- NER ---- #
        with open(f'./data/{args.ids_entity}.json', 'r') as f:
            self.ids_entity_token = json.load(f)

        self.label2idx = {'O':0, 'symptom':1, 'disease':2, 'position':3}

    def __len__(self):

        return self.data['uni_id'].nunique()

    def __getitem__(self, idx):

        sub = self.data.query('uni_id == @idx')
        uni_id = self.id_encoder.inverse_transform([idx])[0]
        entity_dict = self.ids_entity_token[uni_id]

        pr = entity_dict['pred']

        QA = ''.join(('{S'+ sub.questions + ' ' + sub.answers + 'E}').values)
        ETC = [ner for ner in pr]

        if self.transforms is not None:
            tokens, ner_results = self.transforms.forward(QA, ETC, self.tokenizer)

        ner_results = [en.replace('B_', '').replace('I_', '') for en in ner_results]

        tokens = ["[CLS]"] + tokens
        ner_results = ['O'] + ner_results
        if len(tokens) >= 512:
            tokens  = tokens[:512-1]
            ner_results = ner_results[:512-1]

            tokens.append("[SEP]")
            ner_results.append('O')

        input_ids  = self.tokenizer.convert_tokens_to_ids(tokens)

        entity_ids = [self.label2idx[i] for i in ner_results]
        entity_idx = np.where(np.array(entity_ids)!=0)[0]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        label = int(sub['진료과1'].unique()[0])

        return {'label': label,
                'input_ids': input_ids,
                'entity_ids': entity_ids,
                'attention_mask': attention_mask, 
                'token_type_ids': token_type_ids,
                'entity_idx': entity_idx,
                }