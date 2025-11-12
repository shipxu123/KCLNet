import os
import json
import time
import pprint
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from tensorboardX import SummaryWriter

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.autograd import Variable
from torch_geometric.data import Data, Batch, DataLoader

from torch_geometric.utils import add_self_loops

import arl.data.augmentation as A
import arl.loss as L

from arl.utils.misc import makedir, create_logger, get_logger, AverageMeter, load_state_model, load_state_optimizer,\
                         parse_config, set_seed, param_group_all, modify_state, save_load_split, gen_uniform_80_80_20_split
from arl.model import model_entry
from arl.data import GEDDataset
from arl.optimizer import optim_entry
from arl.lr_scheduler import scheduler_entry

import pdb

train_mean = 8.1039
train_std = 4.6664

class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim=128, dropout_rate=0.6):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(self.encoder.n_class * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, data1, data2):
        g  = F.relu(self.encoder(data1))
        g1 = F.relu(self.encoder(data2))

        z = torch.concat([g, g1], dim=1)
        z = F.relu(self.bn1(self.fc1(z)))
        z = self.dropout(z)
        z = F.relu(self.bn2(self.fc2(z)))
        z = self.dropout(z)
        z = self.fc3(z)

        return g, g1, z


class GEDSolver(object):

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_lr_scheduler()
        self.build_data()

    def setup_env(self):
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)

        self.tb_logger = SummaryWriter(self.path.event_path)

        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')

        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
            self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
            if 'last_iter' not in self.state:
                self.state['last_iter'] = 0
            if 'last_epoch' not in self.state:
                self.state['last_epoch'] = -1
        else:
            self.state = {}
            self.state['last_iter'] = 0
            self.state['last_epoch'] = -1

        # # others
        # torch.backends.cudnn.benchmark = True
        self.seed_base: int = int(self.config.seed_base)
        # set seed
        self.seed: int = self.seed_base
        set_seed(seed=self.seed)

    def build_model(self):
        self.model = model_entry(self.config.model)
        
        if getattr(self.config, "pretrain", False):
            modified_state = {}

            # hard code
            if 'model' in self.state:
                modified_state['model'] = {}
                for key in self.state['model']:
                    modified_state['model'][key.lstrip("encoder.")] = self.state['model'][key]
                load_state_model(self.model, modified_state['model'])
            else:
                modified_state = {}
                for key in self.state['model']:
                    modified_state[key.lstrip("encoder.")] = self.state['model'][key]
                load_state_model(self.model, modified_state)
        else:
            if 'model' in self.state:
                load_state_model(self.model, self.state['model'])
            else:
                load_state_model(self.model, self.state)

        # self.model = self.model.cuda()
        # self.model = self.model.eval()
        self.model = Encoder(encoder=self.model).cuda()

    def _build_optimizer(self, opt_config, model):
        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(model, pconfig)
        opt_config.kwargs.params = param_group
        return optim_entry(opt_config)

    def build_optimizer(self):
        self.optimizer = self._build_optimizer(self.config.optimizer, self.model)
        if not getattr(self.config, "pretrain", False):
            if 'optimizer' in self.state:
                load_state_optimizer(self.optimizer, self.state['optimizer'])

    def _build_lr_scheduler(self, lr_config, optimizer):
        lr_config.kwargs.optimizer = optimizer
        if getattr(self.config, "pretrain", False):
            lr_config.kwargs.last_epoch = -1
        else:
            lr_config.kwargs.last_epoch = self.state['last_epoch']
        return scheduler_entry(lr_config)

    def build_lr_scheduler(self):
        self.lr_scheduler = self._build_lr_scheduler(self.config.lr_scheduler, self.optimizer)

    def build_data(self):
        """
        Specific for Iuductive tasks
        """
        if not getattr(self.config.data, 'max_epoch', False):
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.T_max

        train_dataset = GEDDataset(self.config.data.root_path, name='train')
        val_dataset   = GEDDataset(self.config.data.root_path, name='val')
        test_dataset  = GEDDataset(self.config.data.root_path, name='test')

        train_loader = DataLoader(train_dataset, batch_size=self.config.data.train.batch_size, shuffle=self.config.data.train.shuffle, follow_batch=['x', 'x2'])
        val_loader  = DataLoader(val_dataset, batch_size=self.config.data.val.batch_size, shuffle=self.config.data.val.shuffle, follow_batch=['x', 'x2'])
        test_loader = DataLoader(test_dataset, batch_size=self.config.data.test.batch_size, shuffle=self.config.data.test.shuffle, follow_batch=['x', 'x2'])

        self.train_data = {'loader': train_loader}
        self.val_data = {'loader': val_loader}
        self.test_data = {'loader': test_loader}


    def _pre_train(self, model):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top2 = AverageMeter(self.config.saver.print_freq)
        self.meters.top3 = AverageMeter(self.config.saver.print_freq)

        model.train()

        self.num_classes = self.config.model.kwargs.get('n_class', 1000)
        self.topk = [1, 2, 3] if self.num_classes >= 3 else [1, 2]
        self.criterion = torch.nn.MSELoss()

    def _train(self, model):
        self._pre_train(model=model)

        iter_per_epoch = len(self.train_data['loader'])
        total_step = iter_per_epoch * self.config.data.max_epoch
        end = time.time()

        best_prec1_val, best_prec1_test = 0, 0

        for epoch in tqdm(range(0, self.config.data.max_epoch)):
            start_step = epoch * iter_per_epoch

            if start_step < self.state['last_iter']:
                continue

            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_lr()[0]

            for i, data in enumerate(self.train_data['loader']):
                curr_step = start_step + i

                # jumping over trained steps
                if curr_step < self.state['last_iter']:
                    continue

                start_time = time.time()

                # get_data
                inp, target = data, Variable(data.y)
                # inp, target = inp.cuda(), target.cuda()

                # measure data loading time
                self.meters.data_time.update(time.time() - end)

                data1 = Data(x=data.x, edge_index=data.edge_index, depth=data.depth, batch=data.x_batch).cuda()
                data2 = Data(x=data.x2, edge_index=data.edge_index2, depth=data.depth2, batch=data.x2_batch).cuda()

                # forward
                g, g1, logits = model(data1, data2)
                
                # logits = (train_mean

                target = target.cuda().float()
                target = (target - train_mean) / train_std

                # clear gradient
                self.optimizer.zero_grad()

                # compute and update gradient
                loss = self.criterion(logits, target)

                self.meters.losses.reduce_update(loss)

                # clear gradient
                self.optimizer.zero_grad()

                # compute and update gradient
                loss.backward()

                # Clip Grad norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

                # compute and update gradient
                self.optimizer.step()

                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)

                # training logger
                if curr_step % self.config.saver.print_freq == 0:
                    self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                    self.tb_logger.add_scalar('lr', current_lr, curr_step)

                    remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                            f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                            f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                            f'LR {current_lr:.6f}\t' \
                            f'Remaining Time {remain_time} ({finish_time})'
                    self.logger.info(log_msg)

                end = time.time()

            # testing during training
            if curr_step >= 0 and (epoch + 1) % self.config.saver.val_epoch_freq == 0:
                metrics = self._validate(model=model)
                loss_val = metrics['loss']
                mae_val = metrics['mae']
                rmse_val = metrics['rmse']
                r2_val = metrics['r2']

                metrics = self._evaluate(model=model)
                loss_test = metrics['loss']
                mae_test = metrics['mae']
                rmse_test = metrics['rmse']
                r2_test = metrics['r2']

                # testing logger
                self.tb_logger.add_scalar('mae_val', mae_val, curr_step)
                self.tb_logger.add_scalar('rmse_val', rmse_val, curr_step)
                self.tb_logger.add_scalar('r2_val', r2_val, curr_step)

                self.tb_logger.add_scalar('mae_test', mae_test, curr_step)
                self.tb_logger.add_scalar('rmse_test', rmse_test, curr_step)
                self.tb_logger.add_scalar('r2_test', r2_test, curr_step)

                # save ckpt
                if self.config.saver.save_many:
                    ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                else:
                    ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'

                self.state['model'] = model.state_dict()
                self.state['optimizer'] = self.optimizer.state_dict()
                self.state['last_epoch'] = epoch
                self.state['last_iter'] = curr_step
                torch.save(self.state, ckpt_name)


    def _compute_accuracies(self, logits, target):
        # topk = (1, 2, 3) if num_classes >= 3 else (1, 2)
        return [torchmetrics.functional.accuracy(logits, target, num_classes=self.num_classes, top_k=topk) for topk in self.topk]

    @torch.no_grad()
    def _evaluate(self, model):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        auc = AverageMeter(0)
        mae = AverageMeter(0)
        rmse = AverageMeter(0)

        model.eval()
        criterion = torch.nn.MSELoss()
        val_iter = len(self.test_data['loader'])
        end = time.time()

        all_preds = []
        all_targets = []

        # # 收集所有target值
        # targets = []
        # for data in self.test_data['loader']:
        #     targets.extend(data.y.numpy())
        
        # # 计算std和mean
        # targets = np.array(targets)
        # train_mean = np.mean(targets)
        # train_std = np.std(targets)
        
        # print(f"Target mean: {train_mean:.4f}")
        # print(f"Target std: {train_std:.4f}")
        # import sys
        # sys.exit()

        for i, data in enumerate(self.test_data['loader']):
            # get_data
            # inp = data.cuda()
            inp, target = data, Variable(data.y)

            data1 = Data(x=data.x, edge_index=data.edge_index, depth=data.depth, batch=data.x_batch).cuda()
            data2 = Data(x=data.x2, edge_index=data.edge_index2, depth=data.depth2, batch=data.x2_batch).cuda()

            g, g1, logits = model(data1, data2)
            logits = logits * train_std + train_mean

            target = target.cuda().float()

            # measure f1_score and record loss
            loss = criterion(logits, target)

            num = inp.size(0)
            losses.update(loss.item(), num)

            all_preds.append(logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        final_loss = loss_sum.item()/total_num.item()

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate MAE
        mae_value = F.l1_loss(torch.tensor(all_preds), torch.tensor(all_targets)).item()

        # Calculate RMSE
        rmse_value = torch.sqrt(F.mse_loss(torch.tensor(all_preds), torch.tensor(all_targets))).item()

        # Calculate R²
        r2_value = r2_score(all_targets, all_preds)

        # Calculate Spearman's rank correlation coefficient
        spearman_corr, _ = spearmanr(all_targets, all_preds)

        # Calculate Kendall's tau
        kendall_corr, _ = kendalltau(all_targets, all_preds)

        self.logger.info(f' * Loss {final_loss:.3f}\t * \
            MSE {final_loss:.3f}\t * \
            MAE {mae_value:.3f}\t * \
            RMSE {rmse_value:.3f}\t * \
            R^2 {r2_value:.3f}\t * \
            Spearman {spearman_corr:.3f}\t * \
            Kendall {kendall_corr:.3f}\ttotal_num={total_num.item()}')

        model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['mae'] = mae_value
        metrics['rmse'] = rmse_value
        metrics['r2'] = r2_value
        metrics['spearman'] = spearman_corr
        metrics['kendall'] = kendall_corr
        return metrics


    @torch.no_grad()
    def _validate(self, model):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top2 = AverageMeter(0)
        top3 = AverageMeter(0)
        tpr = AverageMeter(0)
        f1 = AverageMeter(0)
        auc = AverageMeter(0)

        model.eval()
        criterion = torch.nn.MSELoss()
        val_iter = len(self.val_data['loader'])
        end = time.time()

        all_preds = []
        all_targets = []

        for i, data in enumerate(self.val_data['loader']):
            # get_data
            inp, target = data, Variable(data.y)

            data1 = Data(x=data.x, edge_index=data.edge_index, depth=data.depth, batch=data.x_batch).cuda()
            data2 = Data(x=data.x2, edge_index=data.edge_index2, depth=data.depth2, batch=data.x2_batch).cuda()

            g, g1, logits = model(data1, data2)
            logits = logits * train_std + train_mean

            target = target.cuda().float()

            # measure f1_score and record loss
            loss = criterion(logits, target)

            num = inp.size(0)
            losses.update(loss.item(), num)

            all_preds.append(logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Val: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        final_loss = loss_sum.item()/total_num.item()

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate MAE
        mae_value = F.l1_loss(torch.tensor(all_preds), torch.tensor(all_targets)).item()

        # Calculate RMSE
        rmse_value = torch.sqrt(F.mse_loss(torch.tensor(all_preds), torch.tensor(all_targets))).item()

        # Calculate R²
        r2_value = r2_score(all_targets, all_preds)

        self.logger.info(f' * Loss {final_loss:.3f}\t * \
            MSE {final_loss:.3f}\t * \
            MAE {mae_value:.3f}\t * \
            RMSE {rmse_value:.3f}\t * \
            R² {r2_value:.3f}\ttotal_num={total_num.item()}')

        model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['mae'] = mae_value
        metrics['rmse'] = rmse_value
        metrics['r2'] = r2_value
        return metrics

    def train(self):
        self._train(model=self.model)

    def evaluate(self):
        self._evaluate(model=self.model)

    def validate(self):
        self._validate(model=self.model)

def main():
    parser = argparse.ArgumentParser(description='Netlist Basic Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train')

    args = parser.parse_args()
    # build solver
    solver = GEDSolver(args.config)

    # evaluate or fintune or train_search
    if args.phase == 'train':
        if getattr(solver.config, "pretrain", False):
            solver.state['last_epoch'] = 0
            solver.state['last_iter'] = 0

        if solver.state['last_epoch'] <= solver.config.data.max_epoch:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_epoch!')
    elif args.phase == 'evaluate':
        solver.evaluate()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()