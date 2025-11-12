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
import torch.nn.functional as F

from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import add_self_loops
from networkx.readwrite import json_graph

import arl.data.augmentation as A
import arl.loss as L

from arl.utils.misc import makedir, create_logger, get_logger, AverageMeter, load_state_model, load_state_optimizer,\
                         parse_config, set_seed, param_group_all, modify_state, save_load_split, gen_uniform_80_80_20_split
from arl.model import model_entry
from arl.data import GraphClsDataset
from arl.optimizer import optim_entry
from arl.lr_scheduler import scheduler_entry

from arl.model.constrastive_encoders import DualBranchContrast, TriBranchContrast
from arl.eval import get_split, SVMEvaluator

import pdb

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


class ConstrastiveSolver(object):

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
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

    def _build_lr_scheduler(self, lr_config, optimizer):
        lr_config.kwargs.optimizer = optimizer
        lr_config.kwargs.last_epoch = self.state['last_epoch']
        return scheduler_entry(lr_config)

    def build_lr_scheduler(self):
        self.lr_scheduler = self._build_lr_scheduler(self.config.lr_scheduler, self.optimizer)

    def build_data(self):
        """
        Specific for Pretrain tasks
        """
        if not getattr(self.config.data, 'max_epoch', False):
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.T_max

        train_dataset = GraphClsDataset(self.config.data.root_path, name='train')
        test_dataset  = GraphClsDataset(self.config.data.root_path, name='test')

        train_loader = DataLoader(train_dataset, batch_size=self.config.data.train.batch_size, shuffle=self.config.data.train.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=self.config.data.test.batch_size, shuffle=self.config.data.test.shuffle)

        self.train_data = {'loader': train_loader}
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
        self.criterion = L.KCLLoss(self.config.loss.L, self.config.loss.topk_neg).to(device)

    def _train(self, model):
        self._pre_train(model=model)
        model.eval()

        iter_per_epoch = len(self.train_data['loader'])
        total_step = iter_per_epoch * self.config.data.max_epoch
        end = time.time()

        for epoch in tqdm(range(0, self.config.data.max_epoch)):
            start_step = epoch * iter_per_epoch

            if start_step < self.state['last_iter']:
                continue

            self.lr_scheduler.step()
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]

            for i, data in enumerate(self.train_data['loader']):
                curr_step = start_step + i

                # jumping over trained steps
                if curr_step < self.state['last_iter']:
                    continue

                start_time = time.time()

                # get_data
                inp, target = data, Variable(data.graph_label)
                inp, target = inp.cuda(), target.cuda()

                self.meters.data_time.update(time.time() - start_time)

                # forward
                logits, I_embeddings = model(inp)

                # clear gradient
                self.optimizer.zero_grad()

                # compute and update gradient
                loss = self.criterion(I_embeddings)

                # compute and update gradient
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
                    self.tb_logger.add_scalar('loss_train', loss.item(), curr_step)

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

            # testing After training
            if curr_step >= 0 and (epoch + 1) % self.config.saver.val_epoch_freq == 0:
                # eval_result = self._evaluate(model=model)

                # test_acc = eval_result['accuray']
                # test_micro_f1 = eval_result['micro_f1']
                # test_macro_f1 = eval_result['macro_f1']

                # # testing logger
                # self.tb_logger.add_scalar('acc1_test', test_acc, curr_step)
                # self.tb_logger.add_scalar('micro_f1_test', test_micro_f1, curr_step)
                # self.tb_logger.add_scalar('macro_f1_test', test_macro_f1, curr_step)

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


    @torch.no_grad()
    def _evaluate(self, model):
        model.eval()

        # get_data
        x = []
        y = []

        for data in self.test_data['loader']:
            # get_data
            inp, target = data, Variable(data.graph_label)
            inp, target = inp.cuda(), target.cuda()

            logits, I_embeddings = model(inp)
            x.append(logits)
            y.append(data.graph_label)

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)


        x, y = x.numpy(), y.numoy()
        split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.15)
        result = SVMEvaluator(linear=True)(x, y, split)

        final_top1     = result['accuray']
        final_macro_f1 = result['micro_f1']
        final_micro_f1 = result['macro_f1']

        self.logger.info(f' * Prec@1 {final_top1:.3f}\t * F1@Macro {final_macro_f1:.3f} \t'\
                         f' * F1@Micro {final_micro_f1:.3f} \ttotal_num={x.size()[0]}')

        model.train()

        return result


    def train(self):
        self._train(model=self.model)

    def evaluate(self):
        self._evaluate(model=self.model)


def main():
    parser = argparse.ArgumentParser(description='Netlist Basic Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train')

    args = parser.parse_args()
    # build solver
    solver = ConstrastiveSolver(args.config)

    # evaluate or fintune or train_search
    if args.phase == 'train':
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