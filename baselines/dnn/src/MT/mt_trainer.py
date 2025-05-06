import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import scipy.special as special
from sklearn import metrics
import net_mt as net
from train_utils import print_and_log, if_mkdir
from pathlib import Path


class Trainer():
    def __init__(self, hypers, dirs):
        super().__init__()
        self.hypers = hypers
        self.model = None 
        self.dirs = dirs
        
    
    def check_model(self):
        if self.model is None:
            print_and_log("build a model first")
            sys.exit()
    
    def build(self):
        kwargs = self.hypers
        
        num_tasks = self.hypers['num_tasks']
        lr = self.hypers['learning_rate']
        num_epochs = self.hypers['epochs']
        device = self.hypers['device']
        
        train_loss_list_dict = {} # len=num_tasks
        val_loss_list_dict = {} # len=num_tasks
        test_loss_dict = {}
        
        train_r2_list_dict = {}   # <-- changed from auc to r2
        val_r2_list_dict = {}
        test_r2_dict = {}
        for tidx in range(num_tasks):
            train_loss_list_dict[tidx] = [] # len=num_epochs
            val_loss_list_dict[tidx] = [] # len=num_epochs
            test_loss_dict[tidx] = None
            train_r2_list_dict[tidx] = []
            val_r2_list_dict[tidx] = []
            test_r2_dict[tidx] = None
            
        self.train_loss_list_dict = train_loss_list_dict
        self.val_loss_list_dict = val_loss_list_dict
        self.test_loss_dict = test_loss_dict
        self.train_r2_list_dict = train_r2_list_dict
        self.val_r2_list_dict = val_r2_list_dict
        self.test_r2_dict = test_r2_dict
        
        self.val_loss_list_dict['min'] = [None] * num_tasks ## container for min loss
        
        task_stop_bool = np.zeros(num_tasks) # len=num_tasks 
        task_stop_bool = task_stop_bool.astype(bool)
        self.task_stop_bool = task_stop_bool
        self.task_stop_dict = {}
        self.task_patiences = np.ones(num_tasks) * -1
        
        self.model = net.Net(**kwargs).to(device)
        self.model_kwargs = kwargs
        
        self.loss_fn = nn.MSELoss()  # <-- changed from CrossEntropyLoss to MSELoss
        self.shared_optimizer = torch.optim.Adam(self.model.shared_layers.parameters(), lr=lr)
        self.shared_optimizer.zero_grad()
        self.tasks_optimizer = torch.optim.Adam(self.model.task_specific_layers.parameters(), lr=lr)
        self.tasks_optimizer.zero_grad()
        self.task_param_list = list(self.model.task_specific_layers.named_parameters()) # [0.weights, 0.bias, 1.weights, 1.bias, ...]

    def freeze_except_training(self, task_idx):
        task_weight_idx = task_idx*2
        task_bias_idx = task_idx*2+1
        for param_idx, named_param in enumerate(self.task_param_list):
            name, param = named_param
            if param_idx == task_weight_idx or param_idx == task_bias_idx:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    
    def train_single_epoch(self, train_loaders):
        self.check_model()
        model = self.model
        model.train()
        device = self.hypers['device']
        num_tasks = self.hypers['num_tasks']
        
        loss_fn = self.loss_fn
        shared_optimizer = self.shared_optimizer
        tasks_optimizer = self.tasks_optimizer
        task_param_list = self.task_param_list
        
        train_loader = train_loaders[0] # len(train_loaders)=1
        running_loss = dict()
        logits_dict = {}
        labels_dict = {}
        for tidx in range(num_tasks):
            running_loss[tidx] = []
            logits_dict[tidx] = np.empty((0,), float)   # <-- 1D for regression
            labels_dict[tidx] = np.empty((0,), float)
        
        for step, batch in enumerate(train_loader):
            inputs_batch = batch['ecfp'].to(device, non_blocking=True)
            labels_batch = batch['label'].to(device, non_blocking=True).float()  # <-- float for regression
            task_idx_batch = batch['task_idx'].to(device, non_blocking=True)
            task_idx = task_idx_batch[0].detach().item()
            
            if self.task_stop_bool[task_idx]:
                continue

            self.freeze_except_training(task_idx)

            outputs_batch = model(inputs_batch.float(), task_idx)
            outputs_batch = outputs_batch.squeeze(1)  # <-- ensure 1D output
            
            # breakpoint()

            loss_batch = loss_fn(outputs_batch.float(), labels_batch.float())
            running_loss[task_idx].append(loss_batch.item())
            
            outputs_batch_np = outputs_batch.detach().cpu().numpy()
            logits_dict[task_idx] = np.append(logits_dict[task_idx], outputs_batch_np, axis=0)
            
            labels_batch_np = labels_batch.detach().cpu().numpy()
            labels_dict[task_idx] = np.append(labels_dict[task_idx], labels_batch_np, axis=0)

            shared_optimizer.zero_grad()
            tasks_optimizer.zero_grad()
            loss_batch.backward()
            shared_optimizer.step()
            tasks_optimizer.step()
            
        for tidx in range(num_tasks):
            
            if self.task_stop_bool[tidx]: # running_loss[tidx] is empty
                continue
                
            ## hold epoch loss
            epoch_loss = np.mean(np.array(running_loss[tidx]))
            self.train_loss_list_dict[tidx].append(epoch_loss)
            
            ## hold epoch r2
            logits = logits_dict[tidx]
            labels = labels_dict[tidx]
            epoch_r2 = self.calculate_r2(logits, labels)
            self.train_r2_list_dict[tidx].append(epoch_r2)

            
    def val_single_epoch(self, val_loaders, epoch):
        self.check_model()
        num_tasks = self.hypers['num_tasks']
        
        test_results = dict()
        for tidx in range(num_tasks):
            val_loader = val_loaders[tidx]
            if self.task_stop_bool[tidx]:
                continue
            self.val_single_model_single_epoch(val_loader, tidx, epoch)

            
    def val_single_model_single_epoch(self, val_loader, tidx, current_epoch):
        self.check_model()
        model = self.model
        model.eval()
        loss_fn = self.loss_fn
        device = self.hypers['device']
        num_tasks = self.hypers['num_tasks']
        
        # create the test result container
        logits = np.empty((0,), float)   # <-- 1D for regression
        labels = np.empty((0,), float)
        
        running_loss = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True).float()
                task_idx_batch = batch['task_idx'].to(device) 
                task_idx = task_idx_batch[0].item() 

                outputs_batch = model(inputs_batch.float(), task_idx)
                outputs_batch = outputs_batch.squeeze(1)
                loss_batch = loss_fn(outputs_batch.float(), labels_batch.float())
                running_loss.append(loss_batch.item())
                
                outputs_batch_np = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch_np, axis=0)
                
                labels_batch_np = labels_batch.cpu().numpy()
                labels = np.append(labels, labels_batch_np, axis=0)
        avg_loss = np.mean(np.array(running_loss))

        if len(self.val_loss_list_dict[tidx]) < 1:
            previous_loss = None
        else:
            previous_loss = self.val_loss_list_dict['min'][tidx] 
        patience = self.task_patiences[tidx]
        epoch_done = current_epoch+1
        
        if previous_loss is None: ## initial epoch
            self.val_loss_list_dict['min'][tidx] = avg_loss 
        else:
            if avg_loss < previous_loss:
                self.task_patiences[tidx] = 0
                self.val_loss_list_dict['min'][tidx] = avg_loss
            else:
                patience += 1
                self.task_patiences[tidx] = patience
                if patience == self.hypers['patience']:
                    self.task_stop_bool[tidx] = True
                    epoch_done = current_epoch+1
                    print_and_log(f'\ttask{tidx} done after {epoch_done} epochs')
                    self.task_stop_dict[epoch_done].append(tidx)
        
            
        ## hold epoch loss
        self.val_loss_list_dict[tidx].append(avg_loss)

        epoch_r2 = self.calculate_r2(logits, labels)
        self.val_r2_list_dict[tidx].append(epoch_r2)
        


    def predict_single(self, test_loader, task_idx) -> dict:
        self.check_model()
        model = self.model
        model.eval()
        loss_fn = self.loss_fn
        device = self.hypers['device']
        num_tasks = self.hypers['num_tasks']
        
        # create the test result container
        logits = np.empty((0,), float)
        labels = np.empty((0,), float)
        
        running_loss = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True).float()
                task_idx_batch = batch['task_idx'].to(device) 
                task_idx = task_idx_batch[0].item() 

                outputs_batch = model(inputs_batch.float(), task_idx)
                outputs_batch = outputs_batch.squeeze(1)
                loss_batch = loss_fn(outputs_batch.float(), labels_batch.float())
                running_loss.append(loss_batch.item())
                
                outputs_batch_np = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch_np, axis=0)
                
                labels_batch_np = labels_batch.cpu().numpy()
                labels = np.append(labels, labels_batch_np, axis=0)
        
        avg_loss = np.mean(np.array(running_loss))
            
        self.test_loss_dict[task_idx] = avg_loss
        
        avg_r2 = self.calculate_r2(logits, labels)
        self.test_r2_dict[task_idx] = avg_r2
        
        self.save_single_predictions(logits, task_idx, teacher=False) 
    
    
    def predict(self, test_loaders: list):
        self.check_model()
        num_tasks = self.hypers['num_tasks']
        
        test_results = dict()
        for tidx in range(num_tasks):
            test_loader = test_loaders[tidx]
            task_result = self.predict_single(test_loader, tidx)


    def teacher_predict_single(self, teacher_loader, tidx):
        self.check_model() 
        model = self.model
        model.eval()
        loss_fn = self.loss_fn
        device = self.hypers['device']
        
        logits = np.empty((0,), float)
        with torch.no_grad():
            for step, batch in enumerate(teacher_loader):
                inputs_batch = batch['ecfp'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True).float()
                task_idx_batch = batch['task_idx'].to(device) 
                task_idx = task_idx_batch[0].item() 

                outputs_batch = model(inputs_batch.float(), task_idx)
                outputs_batch = outputs_batch.squeeze(1)
                outputs_batch_np = outputs_batch.cpu().numpy()
                logits = np.append(logits, outputs_batch_np, axis=0)
        
        self.save_single_predictions(logits, task_idx, teacher=True) 
            

    def teacher_predict(self, teacher_loaders: list):
        self.check_model()
        num_tasks = self.hypers['num_tasks']
        
        test_results = dict()
        for tidx in range(num_tasks):
            test_loader = teacher_loaders[tidx]
            task_result = self.teacher_predict_single(test_loader, tidx)
        
    
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
        
        print_and_log('train')
        for epoch in range(num_epochs):
            print_and_log(f'\tepoch {epoch}')
            
            if np.sum(self.task_stop_bool) == num_tasks:
                print_and_log(f'\tall tasks stop')
                break
                
            self.task_stop_dict[epoch+1] = []
            
            self.train_single_epoch(train_loaders)
            self.val_single_epoch(val_loaders, epoch)
            
            if len(self.task_stop_dict[epoch+1]) == 0:
                pass
            else:
                for task_idx in self.task_stop_dict[epoch+1]:
                    task_test_loader = test_loaders[task_idx]
                    if task_test_loader is not None:
                        self.predict_single(task_test_loader, task_idx)
                    self.task_stop_bool[task_idx] = True
                    
                    # teacher_loader = teacher_loaders[task_idx]
                    # self.teacher_predict_single(teacher_loader, task_idx)  
        
        print_and_log(f'\t{np.sum(self.task_stop_bool == False)} tasks left')
        for task_idx in np.where(self.task_stop_bool == False)[0]:
            task_test_loader = test_loaders[task_idx]
            if task_test_loader is not None:
                self.predict_single(task_test_loader, task_idx)
            self.task_stop_bool[task_idx] = True
            # teacher_loader = teacher_loaders[task_idx]
            # self.teacher_predict_single(teacher_loader, task_idx)
        
        assert np.sum(self.task_stop_bool) == num_tasks, "there are missing tasks"

        print_and_log('save')
        for tidx in range(num_tasks):
            self.save_single_containers(tidx)
            task_train_dataset = train_loaders[0].dataset.taskidx2dataset[tidx].dataset
            # if teacher:
            #     task_teacher_dataset = teacher_loaders[tidx].dataset.dataset
        
        self.save_checkpoints()

    def calculate_r2(self, logits, labels):
        try:
            r2_score = metrics.r2_score(labels, logits)
            return r2_score
        except ValueError:
            return -1
               

    def save_single_predictions(self, predictions, tidx, teacher=False):
        # dname = f'task{tidx:0>2}'
        # task_dir = os.path.join(self.dirs['cluster_dir'], dname)
        # if_mkdir(task_dir)
        # self.dirs['task_dir'][tidx] = task_dir
        cluster_dir = self.dirs['cluster_dir']
        
        ## test logit
        if not teacher: 
             lname = 'test_logit.npy'
        ## teacher logit
        else: 
            lname = 'teacher_logit.npy'
        
        path = os.path.join(cluster_dir, lname)
        np.save(path, predictions)
    
    
    def save_single_containers(self, tidx):
        # dname = f'task{tidx:0>2}'
        # task_dir = os.path.join(self.dirs['cluster_dir'], dname)
        # if_mkdir(task_dir)
        # self.dirs['task_dir'][tidx] = task_dir
        # task_dir = self.dirs['task_dir'][tidx]
        task_dir = self.dirs['cluster_dir']

        train_path = os.path.join(task_dir, f'train_loss.npy')
        np.save(train_path, np.asarray(self.train_loss_list_dict[tidx]))
        train_path = os.path.join(task_dir, f'train_r2.npy')
        np.save(train_path, np.asarray(self.train_r2_list_dict[tidx]))
        
        val_path = os.path.join(task_dir, f'val_loss.npy')
        np.save(val_path, np.asarray(self.val_loss_list_dict[tidx]))
        val_path = os.path.join(task_dir, f'val_r2.npy')
        np.save(val_path, np.asarray(self.val_r2_list_dict[tidx]))
        
        test_path = os.path.join(task_dir, f'test_results.pickle')
        test_result = {'loss': self.test_loss_dict[tidx], 'r2': self.test_r2_dict[tidx]}
        with open(test_path, 'wb') as wf:
            pickle.dump(test_result, wf)
 
    def save_checkpoints(self):
        """
        Save weights of the model in the cluster directory
        Use .pt format
        """
        cluster_dir = self.dirs['cluster_dir']
        checkpoint_path = Path(cluster_dir) / f'weights.pt'
        torch.save(self.model.state_dict(), checkpoint_path)
