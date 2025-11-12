import os
import json
import time
import pprint
import argparse
import datetime
# from tqdm import tqdm
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


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, data):
        aug1, aug2 = self.augmentor
        x, edge_index, batch, edge_attr, node_attr = data.x, data.edge_index, data.batch, data.edge_attr, data.node_label
        x1, edge_index1, edge_weight1, node_attr1 = aug1(x, edge_index, edge_attr, node_attr)
        x2, edge_index2, edge_weight2, node_attr2 = aug2(x, edge_index, edge_attr, node_attr)
        g = self.encoder(data)
        data.x, data.edge_index, data.edge_attr, data.node_label = x1, edge_index1, edge_weight1, node_attr1
        data.x, data.edge_index, data.edge_attr, data.node_label = x2, edge_index2, edge_weight2, node_attr2
        g1 = self.encoder(data)
        g2 = self.encoder(data)
        return g, g1, g2
    
    # def forward(self, data):
    #     aug1, aug2 = self.augmentor
        
    #     # 获取原始数据
    #     x, edge_index = data.x, data.edge_index
    #     edge_attr = getattr(data, 'edge_attr', None)
    #     node_attr = data.node_label
    #     batch = data.batch
    #     # 获取原始表示 - 修改这里
    #     g = self.encoder(x=x, edge_index=edge_index)  # 明确传入参数
    #     # 创建第一个增强版本
    #     x1, edge_index1, edge_weight1, node_attr1 = aug1(x, edge_index, edge_attr, node_attr)
    #     # 直接传入必要的参数，而不是整个data对象
    #     g1 = self.encoder(x=x1, edge_index=edge_index1)
    #     # 创建第二个增强版本
    #     x2, edge_index2, edge_weight2, node_attr2 = aug2(x, edge_index, edge_attr, node_attr)
    #     # 直接传入必要的参数，而不是整个data对象
    #     g2 = self.encoder(x=x2, edge_index=edge_index2)

    #     return g, g1, g2


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
        # self.model = GConv(input_dim=20, hidden_dim=32, num_layers=2).to(device)
        self.model = model_entry(self.config.model).to(device)
    
        aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

        self.model = Encoder(encoder=self.model, augmentor=(aug1, aug2)).to(device)

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

        self.data = {'train_loader': train_loader, 'test_loader': test_loader}

    def _pre_train(self, model):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        model.train()
        # if not getattr(self.config.constrast, 'drop_core', False):
        #     self.criterion = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
        # else:
        self.criterion = TriBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    def _train(self, model):
        self._pre_train(model=model)
        model.eval()

        iter_per_epoch = len(self.data['train_loader'])
        total_step = iter_per_epoch * self.config.data.max_epoch
        end = time.time()

        for epoch in range(0, self.config.data.max_epoch):
            start_step = epoch * iter_per_epoch

            if start_step < self.state['last_iter']:
                continue

            self.lr_scheduler.step()
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]

            for i, data in enumerate(self.data['train_loader']):
                curr_step = start_step + i

                # jumping over trained steps
                if curr_step < self.state['last_iter']:
                    continue

                start_time = time.time()

                data = data.to(device)
                # inp, target = data, Variable(data.y, requires_grad=False)
                self.meters.data_time.update(time.time() - start_time)

                g0, g1, g2 = self.model(data)
                loss = self.criterion(g0=g0, g1=g1, g2=g2)

                self.meters.losses.reduce_update(loss)

                # clear gradient
                self.optimizer.zero_grad()
                # compute and update gradient
                loss.backward()

                # Clip Grad norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                # # compute and update gradient
                self.optimizer.step()

                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)

                # training logger
                if curr_step % 1 == 0:
                    self.tb_logger.add_scalar('loss_train', loss.item(), curr_step)

                    remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                            f'Loss {loss:.3f}\t'\
                            f'Lr {current_lr:.3f}\t'\
                            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
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

        for data in self.data['test_loader']:
            data = data.to(device)
            z, _, _, _, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr, data.node_attr)
            x.append(z)
            y.append(data.y.argmax(axis=1))

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.15)
        result = SVMEvaluator(linear=True)(x, y, split)

        final_top1     = result['accuray']
        final_macro_f1 = result['micro_f1']
        final_micro_f1 = result['macro_f1']

        self.logger.info(f' * Prec@1 {final_top1:.3f}\t * F1@Macro {final_macro_f1:.3f} \t'\
                         f' * F1@Micro {final_micro_f1:.3f} \ttotal_num={x.size()[0]}')

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