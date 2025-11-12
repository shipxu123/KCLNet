import os
import json
import time
import pprint
import argparse
import datetime
from tqdm import tqdm
from easydict import EasyDict
from tensorboardX import SummaryWriter

import torch
import torchmetrics
from torchmetrics import Precision, Recall, F1Score, AveragePrecision, JaccardIndex, AUROC

from torch.autograd import Variable
from torch_geometric.data import DataLoader

from torch_geometric.utils import add_self_loops

from arl.utils.misc import makedir, create_logger, get_logger, AverageMeter, load_state_model, load_state_optimizer,\
                         parse_config, set_seed, param_group_all, modify_state, save_load_split, gen_uniform_80_80_20_split
from arl.model import model_entry
from arl.data import ClusterClsDataset, ClusterClsDatasetOneHot, ClusterClsDatasetFastText
from arl.optimizer import optim_entry
from arl.lr_scheduler import scheduler_entry

import pdb

class MulticlassMetrics:
    def __init__(self, num_classes, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        
        self.precision = Precision(
            task='multiclass', 
            num_classes=num_classes, 
            average='macro'
        ).to(device)
        
        self.recall = Recall(
            task='multiclass',
            num_classes=num_classes,
            average='macro'
        ).to(device)
        
        self.f1 = F1Score(
            task='multiclass',
            num_classes=num_classes,
            average='macro'
        ).to(device)
        
        self.ap = AveragePrecision(
            task='multiclass',
            num_classes=num_classes,
            average=None
        ).to(device)
        
        self.iou = JaccardIndex(
            task='multiclass',
            num_classes=num_classes
        ).to(device)
        
    def update(self, preds, probs, target):
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)
        self.ap.update(probs, target)
        self.iou.update(preds, target)
    
    def compute(self):
        metrics = {
            'precision': self.precision.compute(),
            'recall': self.recall.compute(),
            'f1': self.f1.compute(),
            'AP': self.ap.compute(),
            'mAP': torch.mean(self.ap.compute()),
            'iou': self.iou.compute()
        }
        return metrics
    
    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.ap.reset()
        self.iou.reset()


class ClusterClsSolver(object):

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
            # hard code
            if 'model' in self.state:
                load_state_model(self.model, self.state['model'])
            else:
                load_state_model(self.model, self.state)

        self.model = self.model.cuda()


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

        if getattr(self.config.data, 'type', 'fft') == 'fft':
            train_dataset = ClusterClsDataset(self.config.data.root_path, name='train')
            val_dataset   = ClusterClsDataset(self.config.data.root_path, name='val')
            test_dataset  = ClusterClsDataset(self.config.data.root_path, name='test')
        elif getattr(self.config.data, 'type', 'fft') == 'onehot':
            # train_dataset = ClusterClsDatasetOneHot(self.config.data.root_path, name='train')
            # val_dataset   = ClusterClsDatasetOneHot(self.config.data.root_path, name='val')
            train_dataset = ClusterClsDatasetOneHot(self.config.data.root_path, name='test')
            val_dataset   = ClusterClsDatasetOneHot(self.config.data.root_path, name='test')
            test_dataset  = ClusterClsDatasetOneHot(self.config.data.root_path, name='test')
        else:
            assert(self.config.data.type == 'text')
            train_dataset = ClusterClsDatasetFastText(self.config.data.root_path, name='train')
            val_dataset   = ClusterClsDatasetFastText(self.config.data.root_path, name='val')
            test_dataset  = ClusterClsDatasetFastText(self.config.data.root_path, name='test')

        train_loader = DataLoader(train_dataset, batch_size=self.config.data.train.batch_size, shuffle=self.config.data.train.shuffle)
        val_loader  = DataLoader(val_dataset, batch_size=self.config.data.val.batch_size, shuffle=self.config.data.val.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=self.config.data.test.batch_size, shuffle=self.config.data.test.shuffle)

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
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        model.train()

        self.num_classes = self.config.model.kwargs.get('out_channels', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train(self, model):
        self._pre_train(model=model)
        model.eval()

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
                inp, target = data, Variable(data.cluster_label[data.cluster_label >= 0])
                inp, target = inp.cuda(), target.cuda()

                # measure data loading time
                self.meters.data_time.update(time.time() - end)

                # forward
                logits = model(inp)[data.cluster_label >= 0]

                # clear gradient
                self.optimizer.zero_grad()

                # compute and update gradient
                loss = self.criterion(logits, target)

                acc = torchmetrics.functional.accuracy(logits.cpu(), 
                                                       target.cpu(), 
                                                       task="multiclass", 
                                                       num_classes=self.num_classes)

                self.meters.losses.reduce_update(loss)
                self.meters.top1.reduce_update(acc)

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
                    self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                    self.tb_logger.add_scalar('lr', current_lr, curr_step)

                    remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                            f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                            f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                            f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                            f'LR {current_lr:.6f}\t' \
                            f'Remaining Time {remain_time} ({finish_time})'
                    self.logger.info(log_msg)

                end = time.time()

            # testing during training
            if curr_step >= 0 and (epoch + 1) % self.config.saver.val_epoch_freq == 0:
                # metrics = self._validate(model=model)
                # loss_val = metrics['loss']
                # prec1_val = metrics['top1']

                metrics = self._evaluate(model=model)
                # loss_test = metrics['loss']
                # prec1_test = metrics['mAP@1']

                # # recording best accuracy performance based on validation accuracy
                # if prec1_val > best_prec1_val:
                #     best_prec1_val = prec1_val
                #     best_prec1_test = prec1_test

                self.tb_logger.add_scalar('mAP@1_test', metrics['mAP@1'], curr_step)
                self.tb_logger.add_scalar('recall_test', metrics['recall'], curr_step)
                self.tb_logger.add_scalar('f1_test', metrics['f1'], curr_step)
                self.tb_logger.add_scalar('auc_test', metrics['auc'], curr_step)
                self.tb_logger.add_scalar('IoU_test', metrics['IoU'], curr_step)


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

    def _save_eval_outputs(self, logits, target, save_path="eval_outputs.pt"):
        """
        Save logits, targets, and predicted labels during evaluation into a tensor.
        Format: [logits..., target, pred]
        """
        # logits shape: (batch, num_classes)
        preds = torch.argmax(logits, dim=1)

        # Concatenate: each row = [logits, target, pred]
        combined = torch.cat([logits.cpu(), 
                            target.unsqueeze(1).cpu().float(), 
                            preds.unsqueeze(1).cpu().float()], dim=1)
        
        if os.path.exists(save_path):
            # Accumulate if file already exists
            old = torch.load(save_path)
            combined = torch.cat([old, combined], dim=0)
        
        torch.save(combined, save_path)

    @torch.no_grad()
    def _evaluate(self, model):
        self._pre_train(model=model)

        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)

        task = 'multiclass' if self.num_classes > 2 else 'binary'
        average = 'macro' if self.num_classes > 2 else None

        # Initialize metrics
        recall = Recall(task=task, num_classes=self.num_classes, average=average).to('cuda')
        f1 = F1Score(task=task, num_classes=self.num_classes, average=average).to('cuda')
        ap = AveragePrecision(task=task, num_classes=self.num_classes, average=None).to('cuda')
        iou = JaccardIndex(task=task, num_classes=self.num_classes).to('cuda')
        auroc = AUROC(task=task, num_classes=self.num_classes, average=average).to('cuda')  # Initialize AUROC

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.test_data['loader'])
        end = time.time()

        all_preds = []
        all_targets = []

        for i, data in enumerate(self.test_data['loader']):
            inp, target = data, data.cluster_label[data.cluster_label >= 0]  # Remove Variable for modern PyTorch
            inp, target = inp.cuda(), target.cuda()

            logits = model(inp)[data.cluster_label >= 0]
            loss = criterion(logits, target)

            # Compute predictions and probabilities
            if task == 'multiclass':
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)
            else:  # Binary classification
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

            # Ensure correct shapes
            if task == 'binary' and preds.dim() > 1:
                preds = preds.squeeze()
                probs = probs.squeeze()

            # Update metrics
            recall.update(preds, target)
            f1.update(preds, target)
            ap.update(probs, target)
            iou.update(preds, target)
            auroc.update(probs, target)  # Update AUROC

            # Compute accuracy
            prec1 = torchmetrics.functional.accuracy(logits.cpu(), target.cpu(), 
                                                     task="multiclass",
                                                     num_classes=self.num_classes)

            # Save logits, target, and prediction
            self._save_eval_outputs(logits, target, save_path="eval_outputs.pt")

            num = inp.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # Compute final metrics
        final_recall = recall.compute().item()
        final_f1 = f1.compute().item()
        final_ap = ap.compute()

        # Compute mAP@1, mAP@2, mAP@5
        final_map_at_1 = torch.tensor([torch.nan_to_num(ap_i, nan=0.0).item() for ap_i in final_ap[:1]]).mean().item()

        final_iou = iou.compute().item()
        final_auroc = auroc.compute().item()  # Compute AUROC


        # Log metrics
        self.logger.info(
            f' * mAP {final_map_at_1:.3f}\t'
            # f' * Prec@1 {top1.avg:.3f}\t'
            # f' * Precision {final_precision:.3f}\t'
            f' * Recall {final_recall:.3f}\t'
            f' * F1 {final_f1:.3f}\t'
            f' * AUC {final_auroc}\t'
            f' * IoU {final_iou:.3f}'
        )

        metrics = {
            'loss': losses.avg,
            'mAP@1': final_map_at_1,
            'recall': final_recall,
            'f1': final_f1,
            'auc': final_auroc,
            'IoU': final_iou
        }

        return metrics

    @torch.no_grad()
    def _validate(self, model):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.val_data['loader'])
        end = time.time()

        for i, data in enumerate(self.val_data['loader']):
            # get_data
            inp, target = data, Variable(data.cluster_label[data.cluster_label >= 0])
            inp, target = inp.cuda(), target.cuda()

            logits = model(inp)[data.cluster_label >= 0]

            # measure f1_score and record loss
            loss = criterion(logits, target)

            prec1 = torchmetrics.functional.accuracy(logits.cpu(), 
                                                     target.cpu(), 
                                                     task="multiclass",
                                                     num_classes=self.num_classes)

            num = inp.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])

        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()

        self.logger.info(f' * Prec@1 {final_top1:.3f}\t \
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['top1'] = final_top1
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
    solver = ClusterClsSolver(args.config)

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