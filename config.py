import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import logging
warnings.filterwarnings("ignore")
import time


class Params():
    def free_memory(self):
        for a in dir(self):
            if not a.startswith('__') and hasattr(getattr(self, a), 'free_memory'):
                getattr(self, a).free_memory()

def config():
    cfg = Params()
    cfg.dataset      = 'cora'      # dataset name
    cfg.layers       = 2               # gnn layers
    cfg.h_size       = 128              # hidden_size
    cfg.lr           = 0.001            # learning rate
    cfg.drop_rate    = 0.2             # drop out
    cfg.epoch        = 300             # training epoches
    cfg.bias         = True            # whether using bias
    cfg.cuda         = '1'             # GPU ID
    cfg.T            = 2.0             # task weight soft
    cfg.gnn          = 'gcn'           # gnn backbone
    cfg.task         = 'nlp'           # n:node cls l: link pred p:pairwise
    cfg.act          = F.relu          # activation function
    cfg.split        = [166,83]        # 1000 for training 500 for validation; the rest nodes are used for testing
    cfg.ratio        = 0.5             # training ratio
    cfg.fea_size     = 1433           # cora:1433;citeseer:3703;pubmed:500;reddit:602
    cfg.num_class    = 7               # cora:7;citeseer:6;pubmed:3;reddit:41
    cfg.patience     = 10              # early stopping
    cfg.neg_size     = 1               # neg:pos
    cfg.threshold    = 0.8             # cosine similarity threshold
    cfg.start_es     = 100             # start early stopping
    cfg.current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    cfg.dyn_w        = "1"             # dynamic weight
    cfg.heads        = 4               # multi-heads for GAT
    cfg.attn_drop    = 0.6             # attention droprate for GAT
    cfg.neg_slope    = 0.1             # negative_slope for LeakyReLu in GAT
    cfg.residual     = False           # residual for GAT
    cfg.agg_type     = 'gcn'           # Aggregator type: mean/gcn/pool/lstm
    cfg.mad_calc     = True            # calculate mad
    cfg.subtask_eval = True            # whether evaluate subtasks
    cfg.classifier   = ['lg']          # model for subtask
    cfg.alpha        = 1e-6            # hyper-parameter

    return cfg


def log_config(cfg,logger):
    for key, value in cfg.__dict__.items():
        logger.info('{} : {}'.format(key,value))

cfg = config()