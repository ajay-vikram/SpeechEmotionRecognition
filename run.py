from tqdm import tqdm
import torch
from time import time
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
from typing import Union, Any, Dict, Tuple, List
from utils import AverageMeter
from sklearn.metrics import r2_score



__all__ = ['Runner']


torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)

class Runner:
    def __init__(self, 
                 args: argparse.Namespace) -> None:
        self.args = args
        self.eval_steps = args.eval_steps
        self.criterion = self.get_criterion()

    def get_criterion(self) -> nn.modules.loss._Loss:
        return nn.CrossEntropyLoss()

    def get_model(self, 
                  pretrained: bool, 
                  checkpoint_file: str = None, 
                  verbose: bool = True,
                  w_mul: float = 0.25, 
                  quant: bool = False, 
                  num_classes: int = 2) -> Tuple[nn.Module, Union[Dict[str, Any], None]]:
        if quant:
            model, chpt = qmobilenet(pretrained, checkpoint_file=checkpoint_file, w_mul=w_mul, 
                                     num_output=num_classes, verbose=verbose, 
                                     bias_bit=(23 if (not self.args.stochastic_prune) 
                                               or self.args.sparsity == 0 else 20),
                                     model_stat_dir=self.meta_dir)
        else:
            model, chpt = mobilenet(pretrained, checkpoint_file=checkpoint_file, w_mul=w_mul, 
                                    num_output=num_classes, verbose=verbose)
        return model.to(torch.cuda.current_device(), non_blocking=True), chpt
    
    def get_optimizer(self, 
                      model: nn.Module, 
                      lr: float, 
                      alpha: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                betas = (0.9, 0.999),
                eps=1e-06)
    
    def get_scheduler(self, 
                      optimizer: torch.optim.Optimizer, 
                      steps_per_epoch: int, 
                      total_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.max_lr, steps_per_epoch=steps_per_epoch,
                                                   epochs=total_epochs)
    
    def set_train_dataloader(self, train_dataloader: DataLoader) -> None:
        self.train_dataloader = train_dataloader

    def set_dev_dataloader(self, dev_dataloader: DataLoader) -> None:
        self.dev_dataloader = dev_dataloader

    def set_test_dataloader(self, test_dataloader: DataLoader) -> None:
        self.test_dataloader = test_dataloader

    def set_dataloaders(self, 
                        train_dataloader: DataLoader, 
                        dev_dataloader: DataLoader, 
                        test_dataloader: DataLoader) -> None:
        self.set_train_dataloader(train_dataloader)
        self.set_dev_dataloader(dev_dataloader)
        self.set_test_dataloader(test_dataloader)
    
    def forward(self, 
                X: torch.Tensor, 
                y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(torch.cuda.current_device(), non_blocking=True)
        y = y.to(torch.cuda.current_device(), non_blocking=True)
        pred_y = self.model(X)  
        loss = self.criterion(pred_y, y)
        return pred_y, loss
    
    def run_init(self) -> None:
        self.cur_epoch = 0
        self.prev_loss = 1e9
        self.train_losses, self.dev_losses = AverageMeter(), AverageMeter()
        self.train_loss_history, self.dev_loss_history = [], []

    def train_loop(self) -> float:
        num_correct = 0
        num_samples = 0
        self.model.train()
        epoch_start = time()
        t = tqdm(self.train_dataloader, leave=False, unit="batch")
        for X, y in t:
            self.optimizer.zero_grad(set_to_none=True)
            pred_y, loss = self.forward(X, y)            
            self.train_losses.update(loss.item(), X.shape[0])
            self.train_loss_history.append(self.train_losses.val)
            loss.backward()
            self.optimizer.step() 
            # self.scheduler.step()

            _, predictions = pred_y.max(1)
            target = y
            num_correct += (predictions == target.to('cuda', non_blocking=True)).sum()
            num_samples += predictions.size(0)
                    
            t.set_postfix(accuracy=(num_correct.item() / num_samples), loss=self.train_losses.avg)
        
        epoch_loss = self.train_losses.avg
        epoch_accuracy = 100 * num_correct.item() / num_samples
        epoch_end = time()
        print('Train Loss: {:.4f} | Train Accuracy: {:.4f}% | Time: {:.2f}s'
              .format(epoch_loss, epoch_accuracy, epoch_end - epoch_start))
        self.train_losses.reset()
        return epoch_loss, epoch_accuracy
    
    def dev_loop(self, 
                 fname: str = None) -> float:
        num_correct = 0
        num_samples = 0
        self.model.eval()
        epoch_start = time()
        t = tqdm(self.dev_dataloader, leave=False, unit="batch")
        for X, y in t:
            with torch.no_grad():
                pred_y, loss = self.forward(X, y)
            self.dev_losses.update(loss.item(), X.shape[0])
            self.dev_loss_history.append(self.dev_losses.val)

            _, predictions = pred_y.max(1)
            target = y
            num_correct += (predictions == target.to('cuda', non_blocking=True)).sum()
            num_samples += predictions.size(0)

            t.set_postfix(accuracy=(num_correct.item() / num_samples), loss=self.dev_losses.avg)
        epoch_loss = self.dev_losses.avg
        epoch_accuracy = 100 * num_correct.item() / num_samples
        epoch_end = time()
        print('Val Loss: {:.4f} | Val Accuracy: {:.4f}% | Time: {:.2f}s'
              .format(epoch_loss, epoch_accuracy, epoch_end - epoch_start))
        self.dev_losses.reset()
        if epoch_loss < self.best_loss:
            # self.save_checkpoint(fname)
            print("\033[1;92m \nSaving checkpoint...\n \033[0m")
            self.best_loss = epoch_loss
        return epoch_loss, epoch_accuracy
    
    def _save_checkpoint(self, 
                         checkpoint: Dict[str, Any], 
                         fname: str = None) -> None:
        print("\033[1;92m \nSaving checkpoint...\n \033[0m")
        fname = os.path.join(self.meta_dir, self.args.filename+".pt") \
            if fname is None else fname
            
        torch.save(checkpoint, fname)
        return None
    
    def evaluate(self, 
                 train: bool = False, 
                 dev: bool = False, 
                 test: bool = True) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
        res = []
        out = []
        all_y_true, all_y_pred = [], []
        self.model.eval()
        for phase, dataloader in {'train': self.train_dataloader,
                                  'dev': self.dev_dataloader,
                                  'test': self.test_dataloader}.items():
            if not eval(phase):
                continue
            num_correct = 0
            num_samples = 0
            epoch_start = time()
            y_true, y_pred = [], []
            losses = AverageMeter()
            t = tqdm(dataloader, leave=False, unit="batch")
            for X, y in t:
                with torch.no_grad():
                    pred_y, loss = self.forward(X, y)   
                losses.update(loss.item(), X.shape[0])
                _, predictions = pred_y.max(1)
                target = y
                num_correct += (predictions == target.to('cuda', non_blocking=True)).sum()
                num_samples += predictions.size(0)
                y_pred.append(predictions.cpu().numpy())
                y_true.append(target.cpu().numpy())
                t.set_postfix(loss=losses.avg)

            all_y_true.append(np.concatenate(y_true))
            all_y_pred.append(np.concatenate(y_pred))
                
            out.append((np.concatenate(y_true), np.concatenate(y_pred)))
            acc = 100 * num_correct / num_samples
            epoch_loss = losses.avg
            epoch_end = time()
            print('{} | Epoch loss: {:.4f} | Accuracy: {:.4f}% | Time: {:.2f}s'.format(
                phase, epoch_loss, acc, epoch_end - epoch_start))
            res.append(acc)

        if all_y_true and all_y_pred:
            all_y_true = np.concatenate(all_y_true)
            all_y_pred = np.concatenate(all_y_pred)
        else:
            all_y_true = np.array([])
            all_y_pred = np.array([])
        return all_y_true, all_y_pred
        
    def save_training_loss(self) -> None:
        np.savetxt(os.path.join(self.meta_dir, 'train_loss_history.txt'), 
                    np.stack(self.train_loss_history))
        np.savetxt(os.path.join(self.meta_dir, 'dev_loss_history.txt'), 
                    np.stack(self.dev_loss_history))
        
    def reset_best_loss(self, 
                        val: float = 1e9) -> None:
        self.best_loss = val

    def run_epoch(self, 
                  fname: str = None) -> None:
        print('-' * 10)
        print(f'Epoch: {self.cur_epoch + 1}/{self.epochs}')
        train_epoch_loss, train_epoch_accuracy = self.train_loop()
        val_epoch_loss, val_epoch_accuracy = self.dev_loop(fname)
        print('\n->>lr:{}\n'.format(self.optimizer.param_groups[0]['lr']))
        return None
            