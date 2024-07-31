import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
import numpy as np
import datetime
import json
import pandas as pd
import yaml
import logging
logger=logging.getLogger()
import requests
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from monacobert import *
from omegaconf import DictConfig, OmegaConf, open_dict
import gc 
import time

class Config():
     # model_file_name
    def __init__(self, dbid, cfg_path=None, config=None):
        if config is None:
            # with open(cfgpath, "r") as config_file:
            #     self.config = yaml.load(config_file, Loader=yaml.FullLoader)
            self.config = OmegaConf.load(cfg_path)
        else : self.config=config

        self.model_fn = self.config.data.models_folder
        if not os.path.exists(self.model_fn):
            os.makedirs(self.model_fn)
        self.dataset_dir = os.path.join(self.config.data.raw_folder, f"kt_merged_df-dbid-{dbid}.csv")

        # basic arguments
        self.gpu_id      = torch.device("cuda")  if torch.cuda.is_available() else torch.device('cpu')
        self.train_ratio = .85
        self.valid_ratio = .15
        self.batch_size  = 16
        self.n_epochs    = 51#55
        self.default     = 2
        self.learning_rate = 0.001

        # model, opt, dataset, crit arguments
        self.model_name='monacobert_ctt'
        self.optimizer  = 'adam'
        self.dataset_name = 'assist2017_pid_diff' #dont change
        self.crit = 'binary_cross_entropy'

        # bidkt's arguments
        self.max_seq_len = self.config.model.max_seq_len
        self.num_encoder = self.config.model.num_encoder
        self.hidden_size = self.config.model.hidden_size
        self.num_head    = self.config.model.num_head
        self.output_size = self.config.model.output_size
        self.dropout_p   = self.config.model.dropout_p
        self.use_leakyrelu =self.config.model.use_leakyrelu

        # grad_accumulation
        self.grad_acc:bool=False
        self.grad_acc_iter:int=2



def main_train(args, subject, config=None, cfg_path=None):
    config = Config(dbid=args.dbid, config=config, cfg_path=cfg_path)
    device = config.gpu_id

    # 1. get dataset from loader
    ## 1.1. choose the loaders normal loaders
    if  config.dataset_name == "assist2017_pid_diff":
        dataset = ASSIST2017_PID_DIFF(config.max_seq_len, subject=subject, config=config, dataset_dir=config.dataset_dir)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    else:
        print("Wrong dataset_name was used...")

    ## 1.2. data chunk
    train_size = int( len(dataset) * config.train_ratio)
    test_size = len(dataset) - (train_size)

    train_dataset = Subset(dataset, range( train_size ))
    valid_dataset = test_dataset = Subset(dataset, range( train_size, train_size + test_size ))

    ## 1.3. get DataLoader
    train_loader = DataLoader(train_dataset,batch_size = config.batch_size,shuffle = True,  collate_fn = collate)
    # valid_loader = DataLoader(valid_dataset,batch_size = config.batch_size,shuffle = False, collate_fn = collate)
    valid_loader= test_loader  = DataLoader(test_dataset ,batch_size = config.batch_size,shuffle = False, collate_fn = collate)
    # print(f"dataset information : num skill = {num_q}, num_correct values = {num_r}, num_topics= {num_pid}, num_diff= {num_diff}" )

    # 3. select crits using get_crits
    crit = get_crits(config)


    # 2. select models using get_models & select trainers for models, using get_trainers
    if config.model_name == "monacobert_ctt":
            model = MonaCoBERT_CTT(
                num_q=num_q,
                num_r=num_r,
                num_pid=num_pid,
                num_diff=num_diff,

                hidden_size=config.hidden_size,
                output_size=config.output_size,
                num_head=config.num_head,
                num_encoder=config.num_encoder,
                max_seq_len=config.max_seq_len,
                device=device,
                use_leakyrelu=config.use_leakyrelu,
                dropout_p=config.dropout_p
            ).to(device)

            optimizer = get_optimizers(model, config) #select optimizers using get_optimizers

            trainer = MonaCoBERT_CTT_Trainer(
                model=model,
                optimizer=optimizer,
                n_epochs=config.n_epochs,
                device=device,
                num_q=num_q,
                crit=crit,
                max_seq_len=config.max_seq_len,
                grad_acc=config.grad_acc,
                grad_acc_iter=config.grad_acc_iter
            )
    else:
        raise Exception("Wrong config.model_name was used...")


    # 6. use trainer.train to train the models
    # the result contain train_scores, valid_scores, hightest_valid_score, highest_test_score
    print("\n--------------------------\n#training ne\n--------------------")
    train_scores, valid_scores, highest_valid_score, highest_test_score  = trainer.train(train_loader, valid_loader, test_loader, config, EarlyStopping)


    # model's path         # model save
    if highest_test_score > 0.70:
        model_path = os.path.join( config.model_fn, f"dbid{args.dbid}-subject_{dataset.subject}_{config.max_seq_len}_ktmodel.pth") 
        dataset_cfg = {
            "subject": dataset.subject,
            "q2idx"  : dataset.q2idx,
            "pid2idx": dataset.pid2idx,
            "model_cfg": OmegaConf.to_container(config.config['model'], resolve=True)
        }
        with open(os.path.join( config.model_fn,    f"dbid{args.dbid}-subject_{int(dataset.subject)}_{config.max_seq_len}_ktmodel.json"), "w") as outfile: json.dump(dataset_cfg, outfile)
        
        torch.save({'model': trainer.model.state_dict()}, model_path)
        # If you used python train.py, then this will be start first
        print(highest_test_score, "and save model to -->", model_path)
    
    #clear ram:
    del dataset, trainer, train_dataset, valid_dataset, train_loader, valid_loader 
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune Training")
    # dataset
    parser.add_argument('-s', '--subject-id', type=int)
    parser.add_argument('--dbid', type=str)
    parser.add_argument('-l', '--max-len', type=int)
    parser.add_argument('-M', '--manually', action='store_true', help='using configure from -s -l instead of default script')

    args = parser.parse_args()

    dir_path = os.getcwd()
    cfgpath = os.path.join(dir_path, "cfg/config.yml")
    config = OmegaConf.load(cfgpath)

    worksheet_scores=pd.read_csv(os.path.join( os.getcwd(), config.data.raw_folder,f'kt_merged_df-dbid-{args.dbid}.csv'))
    subjects = worksheet_scores['subject_id'].value_counts()
    subjects = subjects[subjects > 5000]
    #print("available data of subjects : \n", subjects)

    if args.manually:
        if args.subject_id in subjects.index:
            print("----subject :", args.subject_id)
            config.model.max_seq_len = args.max_len
            main_train(args, int(args.subject_id), config=config, cfg_path = cfgpath)
    else:
        for subject in subjects.index:
            for max_seq_len in [100, 200]:
                try:
                    print("----subject :", subject)
                    config.model.max_seq_len = max_seq_len
                    main_train(args, int(subject), config=config, cfg_path = cfgpath)
                    time.sleep(5)
                except Exception as e:
                    e = logger.exception(str(e))
                    print(f"GET ERROR in processing--> {e}")
    time.sleep(5)

