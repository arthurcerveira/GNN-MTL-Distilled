import sys
import logging
import numpy as np
import torch 
import torch.utils.data as td


def create_dataloader_list(inputs, labels, task_indices, batch_size, mode: str) -> dict:
    logging.info(f'current mode: {mode}')
    
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    
    using_inputs = inputs[:]
    using_labels = labels[:][:,task_indices]
        
    num_tasks = len(task_indices)

#     print(f'input shape: {using_inputs.shape}, label shape: {using_labels.shape}')
    logging.info(f'input shape: {using_inputs.shape}, label shape: {using_labels.shape}')
    datasetlist_dict = construct_dataset_list(using_inputs, using_labels, mode=mode) 
    dataset_list = datasetlist_dict.get(mode)
    teacher_dataset_list = datasetlist_dict.get('teacher')

    dataloader_list = list()
    teacher_dataloader_list = list()
    for tidx in range(num_tasks):
        task_dataset = dataset_list[tidx]
        if task_dataset is None:
            task_dataloader = None
        else:
            task_dataloader = td.DataLoader(dataset=task_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
        dataloader_list.append(task_dataloader)
        
        if teacher_dataset_list is not None:
            task_teacherset = teacher_dataset_list[tidx]
            task_teacherloader = td.DataLoader(dataset=task_teacherset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            teacher_dataloader_list.append(task_teacherloader)
        
    if len(teacher_dataloader_list) == 0:
        teacher_dataloader_list = None
    dataloaderlist_dict = {mode: dataloader_list, 'teacher': teacher_dataloader_list}    
    return dataloaderlist_dict


def construct_dataset_list(inputs, labels, mode: str) -> dict:            
    num_mols = labels.shape[0]
    num_tasks = labels.shape[1] 
    logging.info(f'num_mols: {num_mols}, num_tasks: {num_tasks}')
    logging.info(f'\t(positive size, negative size) -> (class weight)') 
    
    total_train = 0
    task_dataset_list = []
    teacher_dataset_list = []
    for tidx in range(num_tasks): 
        task_dataset = SingleTaskDataset(inputs, labels[:,tidx], tidx, mode) 
        class_weights = task_dataset.class_weights
        if len(task_dataset.using_indices) == 0:
            task_subset = None
        else:
            task_subset = td.Subset(dataset=task_dataset, indices=task_dataset.using_indices) # because of negative sampling
        task_dataset_list.append(task_subset)
        
        if task_dataset.teacher_indices is not None:
            teacher_subset = td.Subset(dataset=task_dataset, indices=task_dataset.teacher_indices)
            teacher_dataset_list.append(teacher_subset)
        
        logging.info(f'\ttarget no.{tidx}: ({task_dataset.num_pos}, {task_dataset.num_neg}) '\
                     f'-> {class_weights[1]:.2f} vs. {class_weights[0]:.2f}')
        total_train += task_dataset.num_use

    logging.info(f'-> total train size: {total_train}')
    
    if len(teacher_dataset_list) == 0:
        teacher_dataset_list = None
    task_dataset_dict = {mode: task_dataset_list, 'teacher': teacher_dataset_list}
    return task_dataset_dict


class SingleTaskDataset(td.Dataset):
    def __init__(self, ecfps, labels, tidx, mode):
        self.ecfps = ecfps
        self.labels = labels
        self.tidx = tidx
        self.mode = mode
        
        indices = np.arange(len(self.labels))
        self.pos_indices = indices[self.labels==1.0] 
        self.neg_indices = indices[self.labels==0.0]
        self.using_indices = indices[self.labels!=-1.0]
        self.teacher_indices = None
    
        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)
        self.num_use = len(self.using_indices)
        
        try:
            w_pos = (self.num_neg+self.num_pos) / (2.0*self.num_pos) # like sklaern.compute_class_weight 
            w_neg = (self.num_neg+self.num_pos) / (2.0*self.num_neg)
            self.class_weights = np.array([w_neg, w_pos]) # neg=0, pos=1
        except ZeroDivisionError:
            self.class_weights = np.array([0, 0])
    
        if mode == 'train':
            self.teacher_indices = self.using_indices.copy() # before shuffle
            self.using_indices = np.random.choice(self.using_indices, len(self.using_indices), replace=False) # shuffle  
            
    def __len__(self):
        return len(self.using_indices)

    def __getitem__(self, idx): 
        ecfp = torch.from_numpy(self.ecfps[idx]) 
        label = self.labels[idx]
        return {'ecfp': ecfp, 'label': label, 'task_idx': self.tidx} # ecfp & label: torch.tensor, task_idx: integer

