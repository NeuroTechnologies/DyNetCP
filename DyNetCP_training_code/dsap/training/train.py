import time, os, sys
from collections import defaultdict
import random

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from . import train_utils


class InfiniteDataloader():
    def __init__(self, dataset, batch_size, collate_fn, num_workers, num_steps):
        self.dataset = dataset
        self.num_steps = num_steps
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, collate_fn=collate_fn,
            num_workers=num_workers)
        self.iter = self.dataloader.__iter__()

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return self.num_steps

    def __next__(self):
        self.count += 1
        if self.count == self.num_steps:
            raise StopIteration
        else:
            try:
                data = next(self.iter)
            except StopIteration:
                self.iter = self.dataloader.__iter__()
                data = next(self.iter)
            return data


def train(model, datasets, params, writers):
    train_data = datasets['train']
    if 'val' in datasets:
        val_data = datasets['val']
        train_writer, val_writer = writers
    else:
        val_data = None
        train_writer = writers[0]
    no_gpu = params.get('no_gpu', False)
    batch_size = params['training'].get('batch_size', 1000)
    val_batch_size = params['training'].get('val_batch_size', batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    accumulate_steps = params['training'].get('accumulate_steps', 1)
    num_epochs = params['training'].get('num_epochs', 100)
    val_interval = params['training'].get('val_interval', 1)
    val_start = params['training'].get('val_start', 0)
    clip_grad = params['training'].get('clip_grad', None)
    clip_grad_norm = params['training'].get('clip_grad_norm', None)
    verbose = params['training'].get('verbose', False)
    collate_fn = params.get('collate_fn', None)
    continue_training = params.get('continue_training', False)
    num_workers = params['training'].get('num_data_workers', 0)
    use_jitter_correction = params['data'].get('use_jitter_correction')
    early_stopping_iters = params['training'].get('early_stopping_iters', 0)
    use_warmup = params['training'].get('use_lr_warmup')
    working_dir = params['working_dir']
    print("BATCH SIZE: ",batch_size)
    steps_per_epoch = params['training'].get('steps_per_epoch')
    if steps_per_epoch is not None:
        train_data_loader = InfiniteDataloader(train_data, batch_size,
              collate_fn, num_workers, steps_per_epoch)
    else:
        train_data_loader = DataLoader(train_data, batch_size=batch_size, 
                                   shuffle=True, drop_last=True, 
                                   collate_fn=collate_fn, num_workers=num_workers)
    print("NUM BATCHES: ",len(train_data_loader))
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size,
                                 collate_fn=collate_fn, num_workers=num_workers)

    # Build Optimizers
    lr = params['training']['lr']
    wd = params['training'].get('wd', 0.)
    mom = params['training'].get('mom', 0.)
    
    model_params = [param for param in model.parameters() if param.requires_grad]
    use_lbfgs = False
    if params['training'].get('use_adam', False):
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
    elif params['training'].get('use_adamw', False):
        opt = torch.optim.AdamW(model_params, lr=lr, weight_decay=wd)
    elif params['training'].get('use_lbfgs', False):
        opt = torch.optim.Adagrad(model_params, lr=lr)
        use_lbfgs = True
    else:
        opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom)

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    training_scheduler = train_utils.build_scheduler(opt, params)
    if use_warmup:
        warmup_scheduler = train_utils.build_warmup_scheduler(opt, params)
    else:
        warmup_scheduler = None
    exit=False
    if continue_training:
        print("RESUMING TRAINING")
        try:
            model.load(checkpoint_dir)
            train_params = torch.load(training_path)
            start_epoch = train_params['epoch']
            opt.load_state_dict(train_params['optimizer'])
            training_scheduler.load_state_dict(train_params['scheduler'])
            if warmup_scheduler is not None:
                warmup_scheduler.load_state_dict(train_params['warmup_scheduler'])
            best_val_result = train_params['best_val_result']
            best_val_epoch = train_params['best_val_epoch']
            model.steps = train_params['step']
            if early_stopping_iters > 0 and start_epoch - best_val_epoch > early_stopping_iters:
                print("NO IMPROVEMENT FOR %d ITERATIONS. STOPPING."%early_stopping_iters)
                exit=True
                
        except:
            print("CHECKPOINT NOT FOUND. STARTING FROM BEGINNING")
            start_epoch = 1
            best_val_epoch = -1
            best_val_result = 10000000
            model.steps = 0
        print("STARTING EPOCH: ",start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_result = 10000000
        model.steps = 0
    if exit:
        sys.exit(0)
    
    end = start = 0 
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    for epoch in range(start_epoch, num_epochs+1):
        print("EPOCH", epoch, (end-start))
        if use_jitter_correction:
            print("\tjittering data...")
            train_data.jitter_data()
            if val_data is not None:
                val_data.jitter_data()
        model.train_percent = epoch / num_epochs
        start = time.time() 
        loss_counters = defaultdict(float)
        batch_count = 0
        if verbose:
            iterator = enumerate(train_data_loader)
        else:
            iterator = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for batch_ind, batch in iterator:
            model.train()
            if not no_gpu:
                batch = train_utils.batch2gpu(batch)
            inputs = batch['inputs']
            labels = batch['labels']
            loss_dict = model.loss(inputs, labels)
            loss = loss_dict['loss']

            if loss.dim() == 0:
                batch_count += 1
            else:
                batch_count += loss.size(0)

            loss = loss.mean() / accumulate_steps
            loss.backward()
            for loss_name, loss_val in loss_dict.items():
                loss_counters[loss_name] += loss_val.sum().item()
            if verbose:
                print("\tBATCH %d OF %d: %f"%(batch_ind+1, len(train_data_loader), loss.item()))
            if accumulate_steps == -1 or (batch_ind+1)%accumulate_steps == 0:
                if verbose and accumulate_steps > 0:
                    print("\tUPDATING WEIGHTS")
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(model.parameters(), clip_grad)
                elif clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)        
                if use_lbfgs:
                    def _get_loss():
                        loss_dict = model.loss(inputs, labels)
                        loss = loss_dict['loss']
                        loss = loss.mean() / accumulate_steps
                        return loss
                    opt.step(_get_loss)
                else:
                    opt.step()
                model.steps += 1
                opt.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
                if accumulate_steps > 0 and accumulate_steps > len(train_data_loader) - batch_ind - 1:
                    break
            
        if training_scheduler is not None:
            training_scheduler.step()
        
        if train_writer is not None:
            for loss_name, loss_val in loss_counters.items():
                avg_loss = loss_val / batch_count
                train_writer.add_scalar(loss_name, avg_loss, global_step=epoch)
        if ((epoch+1)%val_interval != 0):
            end = time.time()
            continue
        epoch_train_loss = loss_counters['loss'] / batch_count
        if val_data is None:
            epoch_loss = epoch_train_loss
        else:
            model.eval()
            opt.zero_grad()
            if verbose:
                print("COMPUTING VAL LOSSES")
            loss_counters = defaultdict(float)
            batch_count = 0
            with torch.no_grad():
                if verbose:
                    iterator = enumerate(val_data_loader)
                else:
                    iterator = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
                for batch_ind, batch in iterator:
                    if not no_gpu:
                        batch = train_utils.batch2gpu(batch)
                    inputs = batch['inputs']
                    labels = batch['labels']
                    loss_dict = model.loss(inputs, labels)
                    loss = loss_dict['loss']
                    if loss.dim() == 0:
                        batch_count += 1
                    else:
                        batch_count += loss.size(0)
                    for loss_name, loss_val in loss_dict.items():
                        loss_counters[loss_name] += loss_val.sum().item()
                    if verbose:
                        print("\tVAL BATCH %d of %d: %f"%(batch_ind+1, len(val_data_loader), loss.mean().item()))
            if val_writer is not None:
                for loss_name, loss_val in loss_counters.items():
                    avg_loss = loss_val / batch_count
                    val_writer.add_scalar(loss_name, avg_loss, global_step=epoch)
            epoch_loss = loss_counters['loss'] / batch_count
            epoch_val_loss = epoch_loss
            
        if epoch_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = epoch_loss
            print("BEST VAL RESULT. SAVING MODEL...")
            model.save(best_path)
        elif early_stopping_iters > 0 and epoch - best_val_epoch > early_stopping_iters:
            print("NO IMPROVEMENT FOR %d ITERATIONS. STOPPING."%early_stopping_iters)
            break
        model.save(checkpoint_dir)

        save_dict = {
                    'epoch':epoch+1,
                    'optimizer':opt.state_dict(),
                    'scheduler':training_scheduler.state_dict(),
                    'best_val_result':best_val_result,
                    'best_val_epoch':best_val_epoch,
                    'step':model.steps,
                }
        if warmup_scheduler is not None:
            save_dict['warmup_scheduler'] = warmup_scheduler.state_dict()
        torch.save(save_dict, training_path)
        print("EPOCH %d EVAL: "%epoch)
        print("\tCURRENT TRAIN LOSS: %f"%epoch_train_loss)
        if val_data is None:
            print("\tBEST TRAIN LOSS:    %f" % best_val_result)
            print("\tBEST TRAIN EPOCH:   %d" % best_val_epoch)
        else:
            print("\tCURRENT VAL LOSS: %f"%epoch_val_loss)
            print("\tBEST VAL LOSS:    %f"%best_val_result)
            print("\tBEST VAL EPOCH:   %d"%best_val_epoch)
        end = time.time()