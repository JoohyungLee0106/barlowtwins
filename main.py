# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from pygit2 import Repository
import socket
from datetime import datetime
from resnet import resnet18, resnet34, resnet50, resnet101


parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--data', type=Path, default='/home/chris/storage/imagenet', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='/nfs/thena/chris/equi/ckpts', type=Path, help='path to checkpoint directory')



parser.add_argument( "--dataset", default="IMAGENET", choices=["IMAGENET", "CIFAR100", "CIFAR10"], type=str, help="Dataset")
parser.add_argument('--layer-equiv', default=5, choices=[2, 3, 4, 5], type=int, help='layer number to extract equivariance feature')
parser.add_argument('--equiv-mode', default='False', choices=['contrastive', 'lp', 'cosine', 'equiv_only', 'False'], type=str, help='loss type to learn equivariance')
parser.add_argument( "--p", default=2, choices=[1, 2], type=int, help="p for Lp loss")
parser.add_argument('--num-equiv-proj', default=2, choices=[0, 1, 2], type=int, help='number of linear layers for learning equivariance')

# ImageNet: BYOL(1.5e-6), SupCon(1e-4), SimCLR(1e-6), MoCo(1e-4)
parser.add_argument('--weight-equiv', default=1.0, type=float, help='weight for equivariant loss')
parser.add_argument('--transform-types', type=str, nargs='+', default=['flip', 'scale', 'squeeze'], help='transfroms for equi-variance: dsc, nsd, hd')

parser.add_argument('--scale-param', default=0.5, type=float, help='scale parameter')
parser.add_argument('--squeeze-min', default=0.75, type=float, help='squeeze parameter minimum')
parser.add_argument('--squeeze-max', default=1.0, type=float, help='squeeze parameter maximum')
parser.add_argument('--mask-threshold', default=0.95, type=float, help='mask threshold. zero for no mask')
# scale_param=args.scale_param, squeeze_param_min=args.squeeze_param_min, squeeze_param_max=args.squeeze_param_max, mask_threshold=args.mask_threshold, boundary=args.boundary
parser.add_argument('--boundary', default=2, type=int, help='boundary pixels to exclude from equivariance training')
parser.add_argument('--pushtoken', default='o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95', help='Push Bullet token')

def main():
    args = parser.parse_args()
    if args.equiv_mode == 'False':
        args.equiv_mode = False
        assert args.layer_equiv == 5
    tr=''
    for tt in args.transform_types:
        tr+=tt
    args.exp_name = f'{datetime.today().strftime("%m%d")}_{socket.gethostname()}_{Repository(".").head.shorthand}_BarlowTwins_{args.dataset}_lrw{args.learning_rate_weights}_lrb{args.learning_rate_biases}_el{args.layer_equiv}_{args.equiv_mode}_'\
    +f'p{args.p}_weight_equiv{args.weight_equiv}_tr_{tr}_scale_param_{args.scale_param}_sq_{args.squeeze_min}_{args.squeeze_max}_mask_threshold_{args.mask_threshold}_boundary_{args.boundary}'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)
    if not(os.path.isdir(args.checkpoint_dir)):
        args.checkpoint_dir = args.checkpoint_dir.replace('thena', 'thena/ext01')
        if not(os.path.isdir(args.checkpoint_dir)):
            tempdir = '/mnt/aitrics_ext/ext01/chris/temp'
            if not(os.path.isdir(tempdir)):
                os.mkdir(tempdir)
            args.checkpoint_dir = tempdir


    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    t_overall = time.time()
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    # if args.rank == 0:
    #     args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #     stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    #     print(' '.join(sys.argv))
    #     print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    # if (args.checkpoint_dir / 'checkpoint.pth').is_file():
    #     ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
    #                       map_location='cpu')
    #     start_epoch = ckpt['epoch']
    #     model.load_state_dict(ckpt['model'])
    #     optimizer.load_state_dict(ckpt['optimizer'])
    # else:
    start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    if args.rank == 0:
        t_epoch = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.rank == 0:
            print(f'Epoch: {epoch+1}, Time: {round(time.time() - t_epoch, 3)}')
            t_epoch = time.time()
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        #     if step % args.print_freq == 0:
        #         if args.rank == 0:
        #             stats = dict(epoch=epoch, step=step,
        #                          lr_weights=optimizer.param_groups[0]['lr'],
        #                          lr_biases=optimizer.param_groups[1]['lr'],
        #                          loss=loss.item(),
        #                          time=int(time.time() - start_time))
        #             print(json.dumps(stats))
        #             print(json.dumps(stats), file=stats_file)
        # if args.rank == 0:
        #     # save checkpoint
        #     state = dict(epoch=epoch + 1, model=model.state_dict(),
        #                  optimizer=optimizer.state_dict())
        #     torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   os.path.join(args.checkpoint_dir, f'{args.exp_name}.pth'))        
    
    print(f'{args.exp_name} training took {round(time.time()-t_overall, 3)} sec')

    if args.pushtoken:
        from pushbullet import API
        import socket
        pb = API()
        pb.set_token(args.pushtoken)
        push = pb.send_note('BarlowTwins train finished', f'{socket.gethostname()}')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    FEAT = {'resnet18': {1: 64, 2: 64, 3: 128, 4: 256, 5: 512},
            'resnet34': {1: 64, 2: 64, 3: 128, 4: 256, 5: 512},
            'resnet50': {1: 64, 2: 256, 3: 512, 4: 1024, 5: 2048},
            'resnet101': {1: 64, 2: 256, 3: 512, 4: 1024, 5: 2048}}
    
    STRIDE = {'IMAGENET': {1: 4., 2: 4., 3: 8., 4: 16., 5: 32.},
              'CIFAR100': {1: 1., 2: 1., 3: 2., 4: 4., 5: 8.},
              'CIFAR10': {1: 1., 2: 1., 3: 2., 4: 4., 5: 8.}}
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet50(dataset=args.dataset, layer_num=5, zero_init_residual=True, num_classes=1000 if args.dataset == 'IMAGENET' else 100, equiv_mode=args.equiv_mode)
        
        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector_inv = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


        if not args.equiv_mode:
            self.forward = self.forward_inv

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.boundary = args.boundary
        self.layer_equiv = args.layer_equiv

        dim_equiv_proj=128
        dim_equiv_encoder = BarlowTwins.FEAT['resnet50'][args.layer_equiv]
        if args.equiv_mode:
            layers_equiv = []
            for i in range(args.num_equiv_proj-1):
                layers_equiv.append(nn.Conv2d(dim_equiv_encoder, dim_equiv_encoder, kernel_size=1, bias=False))
                layers_equiv.append(nn.BatchNorm2d(dim_equiv_encoder))
                layers_equiv.append(nn.ReLU(inplace=True))

            if args.num_equiv_proj > 0:
                # Should bias be True?
                layers_equiv.append(nn.Conv2d(dim_equiv_encoder, dim_equiv_proj, kernel_size=1, bias=False))   
                
            self.projector_equiv = nn.Sequential(*layers_equiv)
        
        if args.dataset =='IMAGENET':
            size_x = 224
        elif (args.dataset == 'CIFAR10') or (args.dataset == 'CIFAR100'):
            size_x = 32
        else:            
            raise ValueError('<class GERL> Invalid dataset!')

        size_equiv_encoder = int(size_x // BarlowTwins.STRIDE[args.dataset][args.layer_equiv])
        # N이 확실치 않음
        self.N = args.batch_size
        self.shape_fx = torch.tensor([self.N, dim_equiv_proj, size_equiv_encoder, size_equiv_encoder])
        self.center = torch.tensor([[float(size_equiv_encoder-1.0)/2.0, float(size_equiv_encoder-1.0)/2.0]]).expand(self.N, -1)

        # transform = []
        # transform_types.sort()
        # for transform_type in transform_types:
        #     assert transform_type in transforms_dict.keys()
        #     transform.append(transforms_dict[transform_type])
 
        # # SimCLR, BYOL, BarlowTwins
        # self.aug_equiv = ImageSequential(*transform)
        # self.not_flip = [i for i, t in enumerate(transform_types) if t!='flip']

        # self.mask = torch.ones((N, 1, size_equiv_encoder, size_equiv_encoder), requires_grad=False)
        # 256, 128, 28, 28
    
    def forward_equiv(self, y1, y2):
        z1 = self.projector_equiv(self.backbone.forward_single(y1, layer_forward=self.layer_equiv)[:, :, self.boundary:-self.boundary, self.boundary:-self.boundary])
        z2 = self.projector_equiv(self.backbone.forward_single(y2, layer_forward=self.layer_equiv)[:, :, self.boundary:-self.boundary, self.boundary:-self.boundary])
        feature_equiv_fTx=torch.cat([z1, z2], dim=0)

        for i in self.not_flip:
            self.aug_equiv._params[i].data['forward_input_shape'] = self.shape_fx
            self.aug_equiv._params[i].data['center'] = self.center

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


    def forward_inv(self, y1, y2):
        z1 = self.projector_inv(torch.flatten(self.avgpool(self.backbone.forward_single(y1, layer_forward=5)), 1))
        z2 = self.projector_inv(torch.flatten(self.avgpool(self.backbone.forward_single(y2, layer_forward=5)), 1))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
