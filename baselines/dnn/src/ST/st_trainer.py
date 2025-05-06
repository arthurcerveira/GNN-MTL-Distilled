import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as special
from sklearn import metrics
import net 
from train_utils import print_and_log, if_mkdir


## Define the model
class Trainer():
    def __init__(self, hypers, dirs):
        super().__init__()
        self.hypers = hypers # hyper parameter
        self.model = None 
        self.dirs = dirs # saving directory: model, loss
    
    def check_model(self):
        if self.model is None:
            print_and_log("build a model first")
            sys.exit()
    
    def reset_model(self):
        kwargs = {
            'input_dim': self.hypers['input_dim'],
            'hidden1_dim': self.hypers['hidden1_dim'],
            'hidden2_dim': self.hypers['hidden2_dim'],
            'output_dim': self.hypers['output_dim'],
            'drop_rate': self.hypers['drop_rate']
        }
        lr = self.hypers['learning_rate']
        device = self.hypers['device']

        model = net.Net(**kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.1, 
                                                           patience=5,
                                                           verbose=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loss_list = []
        self.val_loss_list = []
        self.test_loss = None
        self.train_auc_list = []
        self.val_auc_list = []
        self.test_auc = None
        self.stop = False
        self.stop_epoch = None
        self.patience = 0
        self.min_vloss = None
        
    def build(self):
        kwargs = {
            'input_dim': self.hypers['input_dim'],
            'hidden1_dim': self.hypers['hidden1_dim'],
            'hidden2_dim': self.hypers['hidden2_dim'],
            'output_dim': self.hypers['output_dim'],
            'drop_rate': self.hypers['drop_rate']
        }
        
        num_tasks = self.hypers['num_tasks']
        lr = self.hypers['learning_rate']
        num_epochs = self.hypers['epochs']
        device = self.hypers['device']
        
        train_dict = dict()
        val_dict = dict()
        
        model = net.Net(**kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.model = model
        self.optimizer = optimizer
        
        self.train_loss_list = [] # training loss container; reset in reset_model()
        self.val_loss_list = []
        self.test_loss = None 
        # self.train_auc_list = []
        # self.val_auc_list = []
        # self.test_auc = None
        self.train_r2_list = []
        self.val_r2_list = []
        self.test_r2 = None
        self.stop = False
        self.stop_epoch = None
        self.patience = 0
        self.min_vloss = None
        
        # self.loss_fn = nn.CrossEntropyLoss() 
        self.loss_fn = nn.MSELoss()  # Regression loss
        self.num_tasks = num_tasks
        self.counts_stops = 0 # stop whole training it it reaches num_tasks

    def train_epoch_single_model(self, train_loader, tidx):
        self.check_model() 
        model = self.model
        model.train()
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        device = self.hypers['device']
        
        logits = np.empty((0), float)
        labels = np.empty((0,), int)
        
        loss_list = []
        for step, batch in enumerate(train_loader):            
            inputs_batch = batch['ecfp'].to(device, non_blocking=True)
            labels_batch = batch['label'].to(device, non_blocking=True)

            outputs_batch = model(inputs_batch.float())
            outputs_batch = outputs_batch.squeeze(1)

            loss_batch = loss_fn(outputs_batch.float(), labels_batch.float())
            loss_list.append(loss_batch.item())
            
            outputs_batch = outputs_batch.detach().cpu().numpy()
            # breakpoint()
            logits = np.append(logits, outputs_batch, axis=0)
            
            labels_batch = labels_batch.detach().cpu().numpy()
            labels = np.append(labels, labels_batch, axis=0)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        avg_loss = np.mean(np.array(loss_list))
        self.train_loss_list.append(avg_loss)
    
        assert logits.shape[0] == labels.shape[0], f'there should be same: {logits.shape[0]} vs. {labels.shape[0]}'
        r2 = self.calculate_r2(logits, labels)
        self.train_r2_list.append(r2)
        # auc = self.calculate_auc(logits, labels)
        # self.train_auc_list.append(auc)
        
        
    def val_epoch_single_model(self, val_loader, tidx, current_epoch):
        self.check_model() 
        model = self.model
        model.eval()
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        device = self.hypers['device']
        
        logits = np.empty((0,), float)
        labels = np.empty((0,), int)
        
        loss_list = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader):            
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True)

                outputs_batch = model(inputs_batch.float())
                outputs_batch = outputs_batch.squeeze(1)
                loss_batch = loss_fn(outputs_batch, labels_batch)
                loss_list.append(loss_batch.item())

                outputs_batch = outputs_batch.detach().cpu().numpy()
                logits = np.append(logits, outputs_batch, axis=0)

                labels_batch = labels_batch.detach().cpu().numpy()
                labels = np.append(labels, labels_batch, axis=0)
            
        avg_loss = np.mean(np.array(loss_list))
        
        if len(self.val_loss_list) < 1:
            previous_loss = None
        else:
            previous_loss = self.min_vloss
        patience = self.patience
        epoch_done = current_epoch+1
        
        if previous_loss is None:
            self.min_vloss = avg_loss
        else:
            if avg_loss < previous_loss:
                self.min_vloss = avg_loss
                self.patience = 0
            else:
                patience += 1
                self.patience = patience
                if patience == self.hypers['patience']:
                    self.stop = True
                    self.stop_epoch = epoch_done
                    print_and_log(f'\ttask{tidx} done after {epoch_done} epochs')
        
        self.val_loss_list.append(avg_loss)
    
        assert logits.shape[0] == labels.shape[0], f'there should be same: {logits.shape[0]} vs. {labels.shape[0]}'
        r2 = self.calculate_r2(logits, labels)
        self.val_r2_list.append(r2)
       

    def predict_single_model(self, test_loader, tidx):
        self.check_model() 
        model = self.model
        model.eval()
        loss_fn = self.loss_fn
        device = self.hypers['device']
        
        logits = np.empty((0,), float)
        labels = np.empty((0,), int)
        
        loss_list = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True)

                outputs_batch = model(inputs_batch.float()) # (batch_size,2) 
                outputs_batch = outputs_batch.squeeze(1)
                loss_batch = loss_fn(outputs_batch, labels_batch)
                loss_list.append(loss_batch.item())
                
                outputs_batch = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch, axis=0)
                
                labels_batch = labels_batch.cpu().numpy()
                labels = np.append(labels, labels_batch, axis=0)
        
        if not len(loss_list) == 0:
            avg_loss = np.mean(np.array(loss_list))
        else:
            avg_loss = None
        self.test_loss = avg_loss
        
        assert logits.shape[0] == labels.shape[0], f'there should be same: {logits.shape[0]} vs. {labels.shape[0]}'
        r2 = self.calculate_r2(logits, labels)
        self.test_r2 = r2
        
        self.save_predictions(logits, tidx, teacher=False) 

    
    def teacher_predict_single_model(self, teacher_loader, tidx):
        self.check_model() 
        model = self.model
        model.eval()
        loss_fn = self.loss_fn
        device = self.hypers['device']
        
        logits = np.empty((0,), float)
        with torch.no_grad():
            for step, batch in enumerate(teacher_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True)
                
                outputs_batch = model(inputs_batch.float()) # (batch_size,2) 
                outputs_batch = outputs_batch.squeeze(1)
                
                outputs_batch = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch, axis=0)
        
        self.save_predictions(logits, tidx, teacher=True) 
    
    
    def train_and_predict(self, train_loaders: list, val_loaders: list, test_loaders: list, teacher_loaders: list=None):
        self.check_model()
        hypers = self.hypers
        device = hypers['device']
        num_epochs = hypers['epochs']
        num_tasks = hypers['num_tasks']
        
        # if not teacher_loaders is None:
        #     teacher = True
        # else:
        #     teacher = False
        
        print_and_log('train and test')
        for tidx in range(num_tasks):
            print_and_log(f'\ttask {tidx}')
            task_train_loader = train_loaders[tidx]
            task_val_loader = val_loaders[tidx]
            task_test_loader = test_loaders[tidx]
            # if teacher:
            #     task_teacher_loader = teacher_loaders[tidx]

            task_epoch = num_epochs
            for epoch in range(task_epoch):

                if self.stop == True:
                    break
                else:
                    self.train_epoch_single_model(task_train_loader, tidx)
                    self.val_epoch_single_model(task_val_loader, tidx, epoch)

            train_r2 = self.train_r2_list[-1]  # *100
            val_r2 = self.val_r2_list[-1]  # *100
            print_and_log(f'\ttrain r2: {train_r2:.3f}, val r2: {val_r2:.3f}')

            if task_test_loader is not None:
                self.predict_single_model(task_test_loader, tidx) 
                # test_auc = self.test_auc * 100
                test_r2 = self.test_r2  # *100
                # print_and_log(f'\ttest auc: {test_auc:.2f}, test r2: {test_r2:.2f}')
                print_and_log(f'\ttest r2: {test_r2:.3f}')

            # if teacher: # after the last epoch
            #     self.teacher_predict_single_model(task_teacher_loader, tidx) 

            self.save_containers(tidx) ### save loss and auc list (both train and val)
            self.save_checkpoints(tidx)
            self.reset_model() 

    def save_checkpoints(self, tidx):
        """
        Save weights of the model in the cluster directory
        Use .pt format
        """
        cluster_dir = self.dirs['cluster_dir']
        checkpoint_path = cluster_dir / f'weights.pt'
        torch.save(self.model.state_dict(), checkpoint_path)

    def calculate_auc(self, logits, labels):
        try:
            scores = special.softmax(logits, axis=1)
            auc_score = metrics.roc_auc_score(labels, scores[:,1])
            return auc_score
        ### if there is only one class in label
        except ValueError:
            return -1
    
    def calculate_r2(self, logits, labels):
        # breakpoint()
        r2_score = metrics.r2_score(labels, logits)
        return r2_score
    
    def save_predictions(self, predictions, tidx, teacher=False):
        # dname = f'task{tidx:0>2}'
        # task_dir = os.path.join(self.dirs['cluster_dir'], dname)
        # if_mkdir(task_dir)
        # self.dirs['task_dir'] = task_dir
        cluster_dir = self.dirs['cluster_dir']
        
        ## test logit
        if not teacher : 
            lname = 'test_logit.npy'
        ## teacher logit
        else: 
            lname = 'teacher_logit.npy'
            
        path = os.path.join(cluster_dir, lname)
        np.save(path, predictions)
        
        
    def save_containers(self, tidx):
        
        # dname = f'task{tidx:0>2}'
        # task_dir = os.path.join(self.dirs['cluster_dir'], dname)
        # if_mkdir(task_dir)
        # self.dirs['task_dir'] = task_dir

        # task_dir = self.dirs['task_dir']
        task_dir = self.dirs['cluster_dir']
        
        train_path = os.path.join(task_dir, f'train_loss.npy')
        np.save(train_path, np.asarray(self.train_loss_list))
        train_path = os.path.join(task_dir, f'train_r2.npy')
        np.save(train_path, np.asarray(self.train_r2_list))
        
        val_path = os.path.join(task_dir, f'val_loss.npy')
        np.save(val_path, np.asarray(self.val_loss_list))
        val_path = os.path.join(task_dir, f'val_r2.npy')
        np.save(val_path, np.asarray(self.val_r2_list))
        
        test_path = os.path.join(task_dir, f'test_results.pickle')
        test_result = {'loss': self.test_loss, 'r2': self.test_r2}
        with open(test_path, 'wb') as wf:
            pickle.dump(test_result, wf)
       
    
    def save_neg_inds(self, tidx, sampled_train_negs, sampled_teach_negs):
        # task_dir = self.dirs['task_dir']
        task_dir = self.dirs['cluster_dir']
        
        train_path = os.path.join(task_dir, f'train_neg_inds.npy')
        np.save(train_path, sampled_train_negs)
        
        teach_path = os.path.join(task_dir, f'teacher_neg_inds.npy')
        np.save(teach_path, sampled_teach_negs)    
        
        