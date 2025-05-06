import os
from pathlib import Path
import sys
import argparse
import logging
import yaml
import pickle
import random

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset_st as ds
import net as net
from st_trainer import Trainer
from train_utils import set_seed, print_and_log, if_mkdir


def cross_entropy_loss_for_val(logits, labels):
    ''' 
    logits.shape: (batch_size, 2)
    labels.shape: (batch_size,)
    cross entropy loss when label is not an integer 
    loss = - sum(label_0*logP_0 + label_1*logP_1)
    '''
    batch_size = logits.shape[0]
    labels = torch.stack([1-labels, labels], dim=1) # shape: (batch_size, 2)
    log_probs = F.log_softmax(logits, dim=1)
    loss = - torch.sum(labels * log_probs) / batch_size
    return loss   


def dump(data, path):
    '''
    dump data to path
    data should be dictionary type
    '''
    with open(path, 'w') as wf:
        yaml.dump(data, wf)


def main(args, datasets, retrain=True):
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-c', '--config', default='../config.yaml', help='path for config file')
    # parser.add_argument('-bs', '--batch-size', type=int, default=256, help='batch size (default: 256)')
    # parser.add_argument('-ep', '--epochs', type=int, default=50, help='epochs (default: 50)')
    # parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('-t', '--trial', type=int, default=0, help='trial index, 0: do all 3 times (default: 0)')
    # parser.add_argument('-v', '--ver', default=None, help='version; (default: None)')
    # parser.add_argument('--cluster', type=int, default=None, help='select a cluster (default: None)')
    # parser.add_argument('-cf', '--cluster-first', type=int, default=None, help='select the first cluster (default: None)')
    # parser.add_argument('-cl', '--cluster-last', type=int, default=None, help='select the last cluster (default: None)')
    # parser.add_argument('--gpu', type=int, default=0, help='select a gpu (default: 0)')
    # parser.add_argument('-d', '--distance', type=int, default=430, help='cluster cut-off distance (default: 430)')
    # parser.add_argument('-h1', '--hidden1', type=int, default=1024, help='hidden1 nodes (default: 1024)')
    # parser.add_argument('-h2', '--hidden2', type=int, default=128, help='hidden2 nodes (default: 128)')
    # parser.add_argument('-dr', '--drop-rate', type=float, default=0.5, help='drop-out rate (default: 0.5)')
    # parser.add_argument('-s', '--seed', type=int, default=0, help='random seed (default: 0)')
    # parser.add_argument('--patience', type=int, default=5, help='patience (default: 5)')
    
    # args = parser.parse_args()
    
    print(f"ST-{args.epochs}-{args.batch_size}-{args.learning_rate}-{args.drop_rate}-{args.seed}")
    with open(args.config) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
        
    # set seed 
    set_seed(args.seed)
    
    # set device 
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    
    # create a result and log directory
    # if not os.path.exists('log'):
    #     os.mkdir('log')
    # if not os.path.exists('results'):
    #     os.mkdir('results')
    # if args.patience == 5:  # parser.get_default('patience'):
    #     version = f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.drop_rate}_{args.seed}'
    #     resdir = os.path.join('results', f'results{version}')
    #     logdir = os.path.join('log', f'log{version}')
    # else:
    #     version = f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.drop_rate}_{args.seed}_{args.patience}'
    #     resdir = os.path.join('results', f'results{version}')
    #     logdir = os.path.join('log', f'log{version}') 
    # if_mkdir(resdir)
    # if_mkdir(logdir)
    resdir = args.checkpoint
        
    # convert config into dictionary
    opt = vars(args).copy()
    config_path = os.path.join(resdir, f'current_config.yaml')
    opt.pop('cluster_first')
    opt.pop('cluster_last')
    opt.pop('cluster')
    # dump(opt, config_path)
        
    # set the base log file
    # logfname = os.path.join(logdir, f'ST.base.train.log')
    # logging.basicConfig(filename=logfname,
    #                     format='%(asctime)s %(levelname)-8s %(message)s',
    #                     level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', 
    #                     force=True)
    print_and_log("===== start =====")
    print_and_log(f'epochs: {args.epochs} | batch-size: {args.batch_size} | learning-rate: {args.learning_rate}')
    print_and_log(f'hidden1: {args.hidden1} | hidden2: {args.hidden2} | drop-rate: {args.drop_rate}')
    print_and_log(f'result dir: {resdir}')

    # load common data    
    # train_inputs = np.load(config['SEADATA']['TRAIN_INPUT'])
    # train_inputs = np.float32(train_inputs)
    # train_labels = np.load(config['SEADATA']['TRAIN_LABEL'])
    # train_labels = np.int64(train_labels)
    
    # val_inputs = np.load(config['SEADATA']['VAL_INPUT'])
    # val_inputs = np.float32(val_inputs)
    # val_labels = np.load(config['SEADATA']['VAL_LABEL'])
    # val_labels = np.int64(val_labels)
    
    # test_inputs = np.load(config['SEADATA']['TEST_INPUT'])
    # test_inputs = np.float32(test_inputs)
    # test_labels = np.load(config['SEADATA']['TEST_LABEL'])
    # test_labels = np.int64(test_labels)
    
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, targets = datasets

    print_and_log(f'train input shape: {train_inputs.shape}, train label shape: {train_labels.shape}')
    print_and_log(f'val input shape: {val_inputs.shape}, val label shape: {val_labels.shape}')
    print_and_log(f'test input shape: {test_inputs.shape}, test label shape: {test_labels.shape}')
    
    # with open(config['SEADATA']['CLUSTER'], 'rb') as f:
    #     cluster_tasks = pickle.load(f)

    hypers = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'input_dim': 2048,
        'hidden1_dim': args.hidden1,
        'hidden2_dim': args.hidden2,
        'output_dim': 1,  # 2 for classification, 1 for regression
        'device': device,
        'drop_rate': args.drop_rate,
        'patience': args.patience
    }
    batch_size = hypers['batch_size']
    num_epochs = hypers['epochs'] 

    # --------- base condition is identical regardless of clusters --------- #

    # train and test the model on every cluster
    # if args.cluster_first is None:
    #     clstr1 = 0
    # else:
    #     clstr1 = args.cluster_first
    # if args.cluster_last is None:
    #     clstr2 = len(cluster_tasks)-1
    # else:
    #     clstr2 = args.cluster_last
    # print_and_log(f'cluster_first: {clstr1}, cluster_last: {clstr2}')
    # for clstr in range(clstr1, clstr2+1):
    for task_idx, target in enumerate(targets):
        # breakpoint()
        # set seed (default: 0)
        set_seed(args.seed)
        
        # get the cluster
        # task_indices = cluster_tasks[clstr]['tinds']
        # num_tasks = len(task_indices)
        num_tasks = 1
        hypers['num_tasks'] = num_tasks
        
        # cname = f'cluster{clstr:0>3}'
        cname = target
        cluster_dir = Path(resdir) / cname
        cluster_dir.mkdir(parents=True, exist_ok=True)
        

        dirs = {
            'cluster_dir': cluster_dir,
            # 'cluster': target,
        }
        
        # set the cluster specific log file
        # logfname = os.path.join(logdir, f'ST.c{clstr:0>3}.train.log')
        # logging.basicConfig(filename=logfname,
        #                     format='%(asctime)s %(levelname)-8s %(message)s',
        #                     level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', 
                            # force=True)
        print_and_log("===== start =====")
        print_and_log(f'epochs: {args.epochs} | batch-size: {args.batch_size} | learning-rate: {args.learning_rate}')
        print_and_log(f'hidden1: {args.hidden1} | hidden2: {args.hidden2} | drop-rate: {args.drop_rate}')
        print_and_log(f'result dir: {cluster_dir.resolve()}')

        if not retrain:
            if Path(cluster_dir / 'weights.pt').exists():
                print_and_log(f'{(cluster_dir / 'weights.pt').resolve()} exists, skip training')
                continue

        # print_and_log(f'current cluster: {clstr} | num tasks: {num_tasks}')
                        
        # generate data loaders
        print_and_log('generate train dataloader')
        train_dataloader_dict = ds.create_dataloader_list(
            train_inputs, train_labels, [task_idx], batch_size, mode='train')
        train_dataloader_list = train_dataloader_dict['train']
        teacher_dataloader_list = train_dataloader_dict['teacher']
        print_and_log(f'train dataloader #: {len(train_dataloader_list)}')
        print_and_log(f'teacher dataloader #: {len(teacher_dataloader_list)}')

        print_and_log('generate val dataloader')
        val_dataloader_dict = ds.create_dataloader_list(
            val_inputs, val_labels, [task_idx], batch_size, mode='val')
        val_dataloader_list = val_dataloader_dict['val']
        print_and_log(f'val dataloader #: {len(val_dataloader_list)}')
        
        print_and_log('generate test dataloader')
        test_dataloader_dict = ds.create_dataloader_list(
            test_inputs, test_labels, [task_idx], batch_size, mode='test')
        test_dataloader_list = test_dataloader_dict['test']
        print_and_log(f'test dataloader #: {len(test_dataloader_list)}')
            
        # build model, train, save
        trainer = Trainer(hypers, dirs)
        trainer.build()
 
        print_and_log('train model and save prediction')
        trainer.train_and_predict(train_dataloader_list, val_dataloader_list, test_dataloader_list, teacher_dataloader_list) # self.results


if __name__ == '__main__':
    main() 