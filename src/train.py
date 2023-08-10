import os
import utils
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from utils import *
from data_modules.dataloader import *
from model.load_model import get_model, get_tokenizer

class Trainer:
    def __init__(self, args, save_path, run):
        super(Trainer, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.run = run

        # Logging
        self.save_path = save_path
        log_file = os.path.join(save_path, 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)

        # Train, Valid Set Split
        df = pd.read_csv(f'./data/{args.file_name}.csv')
        df['questions'] = df['questions'].astype('str').apply(lambda x: x.strip())
        df['answers']   = df['answers'].astype('str').apply(lambda x: x.strip())

        lst = ['소화기내과_상부위장관파트(UGI)', '소화기내과_하부위장관파트(LGI)', '소화기내과_췌담도파트(PB)', '대장항문외과']
        df = df.query('진료과1 in @lst').reset_index(drop=True)
        df['type_label'] = df['type'] + ' ' + df['진료과1']
        
        self.di_encoder = LabelEncoder()
        df['진료과1'] = self.di_encoder.fit_transform(df['진료과1'])
        df_uni = df.drop_duplicates(subset=['uni_id']).reset_index(drop=True)

        print('Data Size : ', df_uni.shape)   

        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df_uni['진료과1']), y=df_uni['진료과1'])
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(self.device))

        kf = StratifiedKFold(n_splits=self.args.Kfold, shuffle=True, random_state=self.args.seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(df_uni, df_uni['진료과1'])):
            df_uni.loc[test_idx, 'fold'] = fold

        # Tokenizer & Network 
        self.tokenizer = get_tokenizer()

        self.df = df
        self.df_uni = df_uni 

    # optimizer, loss, scheduler
    def set_model(self):

        model = get_model().to(self.device)
        model.resize_token_embeddings(len(self.tokenizer.vocab))

        return model
    
    def set_sche_optim(self, model, train_loader):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not 'embeddings' in n], 'lr': self.args.lr, 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not 'embeddings' in n], 'lr': self.args.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'embeddings' in n], 'lr': self.args.lr*10, 'weight_decay': self.args.weight_decay}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters, eps=1e-8)

        t_total = len(train_loader) * self.args.epochs
        warmup_step = int(t_total * 0.1)
        if self.args.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
        elif self.args.scheduler == 'cos':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        return scheduler, optimizer

    # for outer fold of nested CV
    def outer_loop(self):

        test_results = []
        for out_fold in range(5):
            idx_test = self.df_uni.loc[self.df_uni['fold']==out_fold, 'uni_id'].values
            df_train = self.df.query(f'uni_id not in @idx_test').reset_index(drop=True)
            df_test = self.df.query(f'uni_id in @idx_test').reset_index(drop=True)
            print('Testset 라벨 분포\n', self.df_uni.loc[self.df_uni['fold']==out_fold, '진료과1'].value_counts())

            # DataLoader
            self.test_loader = get_loader(self.args, df_test, tokenizer=self.tokenizer, phase='valid', batch_size=self.args.batch_size)
            print('Test  Dataset size: ', {len(self.test_loader.dataset)})

            fold_results = self.inner_loop(out_fold, df_train, self.test_loader)
            test_results.append(fold_results)
            
        t_acc, t_pr, t_re, t_f1, t_mcc = np.mean(test_results, axis=0)
        
        if self.args.logging == True:
            self.run[f'mean_test acc'].append(t_acc)
            self.run[f'mean_test pre'].append(t_pr)
            self.run[f'mean_test rec'].append(t_re)
            self.run[f'mean_test f1'].append(t_f1)
            self.run[f'mean_test mcc'].append(t_mcc)

    # for inner fold of nested CV
    def inner_loop(self, out_fold, df_train, test_loader):

        tr_uni = df_train.drop_duplicates(subset=['uni_id']).reset_index(drop=True)
        print('Data Size : ', tr_uni.shape)
    
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.args.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(tr_uni, tr_uni['진료과1'])):
            tr_uni.loc[val_idx, 'fold'] = fold

        in_folds = list(range(3))
        val_f1_scores = []
        for in_fold in in_folds:
            self.logger.info(f'\n########################################### Train out_{out_fold}/in_{in_fold} fold ###########################################')
            idx_valid = tr_uni.loc[tr_uni['fold']==in_fold, 'uni_id'].values
            df_tr = df_train.query(f'uni_id not in @idx_valid').reset_index(drop=True)
            df_val = df_train.query(f'uni_id in @idx_valid').reset_index(drop=True)
            
            model = self.set_model()
            fix_seed(self.args.seed)
            
            train_loader = get_loader(self.args, df_tr, tokenizer=self.tokenizer, phase='train', batch_size=self.args.batch_size)
            valid_loader = get_loader(self.args, df_val, tokenizer=self.tokenizer, phase='valid', batch_size=self.args.batch_size)

            print('Trainset 라벨 분포\n', tr_uni.loc[tr_uni['fold']!=in_fold, '진료과1'].value_counts())
            print('Test  Dataset size: ', {len(train_loader.dataset)})
            print('Validset 라벨 분포\n', tr_uni.loc[tr_uni['fold']==in_fold, '진료과1'].value_counts())
            print('Test  Dataset size: ', {len(valid_loader.dataset)})

            scheduler, optimizer = self.set_sche_optim(model, train_loader)

            best_f1 = self.train(out_fold, in_fold, train_loader, valid_loader, model, scheduler, optimizer)
            val_f1_scores.append(best_f1)
        
        best_fold = np.argmax(val_f1_scores)
        self.logger.info(f'\nBest f1 score in {best_fold}fold')
        t_acc, t_pr, t_re, t_f1, t_mcc = self.test(out_fold, best_fold, test_loader)

        in_folds.remove(best_fold)
        for f in in_folds:
            os.remove(os.path.join(self.save_path, f'best_model_out{out_fold}_in{f}.pth'))

        return [t_acc, t_pr, t_re, t_f1, t_mcc]

    def train(self, out_fold, in_fold, train_loader, valid_loader, model, scheduler, optimizer):

        # Train / Validate
        best_loss = np.inf
        best_f1 = 0

        for epoch in range(1, self.args.epochs+1):
            self.logger.info(f'Epoch:[{epoch:03d}/{self.args.epochs:03d}]')
            self.epoch = epoch

            # Training
            train_loss, train_acc, train_pre, train_rec, train_f1, train_mcc = self.train_one_epoch(train_loader, model, scheduler, optimizer)
            val_loss, val_acc, val_pre, val_rec, val_f1, val_mcc = self.validate(mode='train', data_loader=valid_loader, model=model)
            
            if self.args.logging == True:
                self.run[f'out_{out_fold}/in_{in_fold}/train loss'].append(train_loss)
                self.run[f'out_{out_fold}/in_{in_fold}/train acc'].append(train_acc)
                self.run[f'out_{out_fold}/in_{in_fold}/train pre'].append(train_pre)
                self.run[f'out_{out_fold}/in_{in_fold}/train rec'].append(train_rec)
                self.run[f'out_{out_fold}/in_{in_fold}/train f1'].append(train_f1)
                self.run[f'out_{out_fold}/in_{in_fold}/train mcc'].append(train_mcc)
                self.run[f'out_{out_fold}/in_{in_fold}/val loss'].append(val_loss)
                self.run[f'out_{out_fold}/in_{in_fold}/val acc'].append(val_acc)
                self.run[f'out_{out_fold}/in_{in_fold}/val pre'].append(val_pre)
                self.run[f'out_{out_fold}/in_{in_fold}/val rec'].append(val_rec)
                self.run[f'out_{out_fold}/in_{in_fold}/val f1'].append(val_f1)
                self.run[f'out_{out_fold}/in_{in_fold}/val mcc'].append(val_mcc)

            # Save models
            if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss):
                best_epoch = epoch
                best_f1 = val_f1
                best_loss = val_loss
                
                # Model weight in Multi_GPU or Single GPU
                state_dict = model.module.state_dict() if self.args.multi_gpu else model.state_dict()
                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                    }, os.path.join(self.save_path, f'best_model_out{out_fold}_in{in_fold}.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------') 
                patience = 0  
            else:
                patience +=1

            if patience == self.args.patience:
                break

        return best_f1

    def train_one_epoch(self, train_loader, model, scheduler, optimizer):

        acc = 0
        preds_list=[]; targets_list=[]
        
        model.train()
        train_loss = utils.AvgMeter()

        for batch in tqdm(train_loader):
            labels, input_ids, entity_ids, attention_mask, token_type_ids, local_attn_mask = batch

            output = model(
                            input_ids  = input_ids,
                            entity_ids = entity_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            local_attention_mask = local_attn_mask,
                            )          
            
            preds = F.softmax(output, dim=1)
            loss = self.criterion(output, labels)
            loss.backward()

            # Gradient Clipping
            if self.args.clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clipping)

            if self.epoch > self.args.warm_epoch:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

            # log
            train_loss.update(loss.mean().item(), n=input_ids.size(0))
            acc += (preds.argmax(dim=1) == labels).sum().item()
            
            preds_list.extend(preds.argmax(dim=1).cpu().detach())
            targets_list.extend(labels.cpu().detach())

        acc /= len(train_loader.dataset)
        pr, re, f1, _  = precision_recall_fscore_support(np.array(targets_list), np.array(preds_list), average='macro')
        mcc  = matthews_corrcoef(np.array(targets_list), np.array(preds_list))

        return train_loss.avg, acc, pr, re, f1, mcc

    # Validation or Dev
    def validate(self, mode, data_loader, model):

        acc = 0
        preds_list=[]; targets_list=[]
        model.eval()

        with torch.no_grad():
            val_loss = utils.AvgMeter()

            for batch in data_loader:
                labels, input_ids, entity_ids, attention_mask, token_type_ids, local_attn_mask = batch

                output = model(
                                input_ids = input_ids,
                                entity_ids = entity_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                local_attention_mask = local_attn_mask
                                )

                preds = F.softmax(output, dim=1)
                loss = self.criterion(output, labels)

                # log
                val_loss.update(loss.mean().item(), n=input_ids.size(0))
                acc += (preds.argmax(dim=1) == labels).sum().item()

                preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
                targets_list.extend(labels.cpu().detach().numpy())

            acc /= len(data_loader.dataset)
            pr, re, f1, _  = precision_recall_fscore_support(np.array(targets_list), np.array(preds_list), average='macro')
            mcc  = matthews_corrcoef(np.array(targets_list), np.array(preds_list))

        if mode == 'train':
            return val_loss.avg, acc, pr, re, f1, mcc
        else:
            return acc, pr, re, f1, mcc
        
    # Test
    def test(self, out_fold, in_fold, test_loader):

        self.logger.info('Evaluation mode on Testset')
        model = self.set_model()
        model.eval()

        best_model = torch.load(os.path.join(self.save_path, f'best_model_out{out_fold}_in{in_fold}.pth'))
        model.load_state_dict(best_model['state_dict'])
        best_epoch = best_model['epoch']

        self.logger.info(f'Best checkpoint in {best_epoch} th epoch.')
        t_acc, t_pr, t_re, t_f1, t_mcc = self.validate(mode='test', data_loader=test_loader, model=model)

        if self.args.logging == True:
            self.run[f'out_{out_fold}/test acc'].append(t_acc)
            self.run[f'out_{out_fold}/test pre'].append(t_pr)
            self.run[f'out_{out_fold}/test rec'].append(t_re)
            self.run[f'out_{out_fold}/test f1'].append(t_f1)
            self.run[f'out_{out_fold}/test mcc'].append(t_mcc)

        return t_acc, t_pr, t_re, t_f1, t_mcc