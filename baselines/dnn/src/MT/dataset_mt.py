import random
import numpy as np
import torch 
import torch.utils.data as td
import logging
import itertools


def create_dataloader_list(inputs, labels, task_indices, batch_size, mode: str) -> dict:
    logging.info(f' current mode: {mode}')
    
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    
    using_inputs = inputs[:]
    using_labels = labels[:][:,task_indices]
    
    num_tasks = len(task_indices)

#     print(f' input shape: {using_inputs.shape}, label shape: {using_labels.shape}')
    logging.info(f' input shape: {using_inputs.shape}, label shape: {using_labels.shape}')
    
    datasetlist_dict = construct_dataset_list(using_inputs, using_labels, mode=mode)
    task_dataset_list = datasetlist_dict.get(mode)
    teacher_dataset_list = datasetlist_dict.get('teacher') 
    

    if mode == 'train': # 
        dataloader_list = list() # length=1 (train) 
        teacher_dataloader_list = list() # lenght=num_tasks (teacher)
        dataset = MultiTaskDataset(task_dataset_list)
        batchsampler = MultiTaskBatchSampler(task_dataset_list, batch_size, shuffle=shuffle)
        dataloader = td.DataLoader(dataset=dataset, batch_sampler=batchsampler, num_workers=2, pin_memory=True)
        dataloader_list.append(dataloader) 
        
        if teacher_dataset_list is not None:
            for tidx in range(num_tasks):
                task_teacherset = teacher_dataset_list[tidx]
                task_teacherloader = td.DataLoader(dataset=task_teacherset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                teacher_dataloader_list.append(task_teacherloader) 
    else:
        dataloader_list = list() # length=num_tasks (test)
        teacher_dataloader_list = list() # length=0 
        for tidx in range(num_tasks):
            task_dataset = task_dataset_list[tidx]
            if task_dataset is None:
                task_dataloader = None
            else:
                test_dataset = task_dataset
                task_dataloader = td.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
            dataloader_list.append(task_dataloader)
    
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


# MultiTaskDataset and MultiTaskBatchSampler are 
# from https://github.com/namisan/mt-dnn with some modifications
class MultiTaskDataset(td.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.taskidx2dataset = dict(zip(range(len(dataset_list)), dataset_list))

    def __len__(self):
        return sum(len(dataset) for dataset in self.dataset_list)

    def __getitem__(self, idx):
        task_idx, sample_idx = idx
        item = self.taskidx2dataset[task_idx][sample_idx] 
        return item    


class MultiTaskBatchSampler(td.BatchSampler):
    def __init__(self, dataset_list, batch_size, shuffle=True, drop_last=False):
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # generate shuffled index batches from each task
        multi_task_batch_idx_list = [] 
        for dataset in dataset_list:
            single_task_batch_idx = self._get_shuffled_index_batches(len(dataset), self.batch_size, self.shuffle) 
            multi_task_batch_idx_list.append(single_task_batch_idx)
        self.batch_idx_list = multi_task_batch_idx_list # length=num_tasks
    
    @staticmethod 
    def _get_shuffled_index_batches(dataset_len, batch_size, shuffle):
        index_batches = [range(i, min(i+batch_size, dataset_len)) for i in range(0, dataset_len, batch_size)] 
        if shuffle:
            random.shuffle(index_batches)
        return index_batches
    
    def __len__(self): # total number of batches
        return sum(len(single_task_batches) for single_task_batches in self.batch_idx_list)
    
    def __iter__(self):
        all_task_iters = [] # list of single task batch generator (len=total number of tasks)
        all_task_indices = [] # list for task selection (len=total number of batches)
        all_task_indices_sep = [] # (len=total_number of tasks)
        
        for single_task_batches in self.batch_idx_list:
            single_task_iters = iter(single_task_batches)
            all_task_iters.append(single_task_iters)
        
        for task_idx in range(len(self.batch_idx_list)):
            single_task_batch_idx = self.batch_idx_list[task_idx]
            task_indices = [task_idx] * len(single_task_batch_idx) # task_idx * the number of batches in single task
            all_task_indices.extend(task_indices)
            all_task_indices_sep.append(task_indices)
        
        if self.shuffle:
            random.shuffle(all_task_indices)
        
        for task_idx in all_task_indices:
            batch = next(all_task_iters[task_idx])
            
            if not self.drop_last:
                yield [(task_idx, sample_idx) for sample_idx in batch]
            else:
                if len(batch) == self.batch_size:
                    yield [(task_idx, sample_idx) for sample_idx in batch] 
                

class SingleTaskDataset(td.Dataset):
    def __init__(self, ecfps, labels, tidx, mode):
        self.ecfps = ecfps
        self.labels = labels
        self.tidx = tidx
        self.mode = mode
        self.add_unk = False
        
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
