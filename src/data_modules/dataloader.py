import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data_modules.dataset import DiagDataset, Augment
from data_modules.collator import Collator

def get_loader(args, df, tokenizer, phase:str, batch_size):

    if phase=='train':
        aug = Augment(args, mode='train', etc=True)

        dataset = DiagDataset(args, df, tokenizer, aug)
        collate_fn = Collator(args)
        data_loader = DataLoader(dataset, 
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                shuffle=True,
                                drop_last=True)
    else:
        aug = Augment(args, mode='valid', etc=True)

        dataset = DiagDataset(args, df, tokenizer, aug)
        collate_fn = Collator(args)
        data_loader = DataLoader(dataset, 
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                shuffle=False,
                                drop_last=False)

    return data_loader
