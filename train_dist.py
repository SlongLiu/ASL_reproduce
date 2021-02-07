import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
import os, sys
import os.path as osp
# import logging
import json
from tqdm import tqdm
import numpy as np

from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, sl_mAP
from src.helper_functions.logger import setup_logger
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
# torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='/data/shilong/data/pretrained/tresnet_l_448.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
# parser.add_argument('--thre', default=0.8, type=float,
#                     metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--distributed', action='store_true', help='using dataparallel')
parser.add_argument('--dtgfl', action='store_true', 
            help='using disable_torch_grad_focal_loss in ASL loss')
parser.add_argument('--output', metavar='DIR', 
                    help='path to output folder')

# distribution training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://127.0.0.1:1234', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    # setup dist training
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # setup logger
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Coco")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        os.makedirs(osp.join(args.output, 'tmpdata'), exist_ok=True)

    # Setup model
    logger.info('creating model...')
    model = create_model(args).cuda()
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    logger.info('done\n')

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path_val = args.data
    data_path_train = args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    logger.info("len(val_dataset)): {}".format(len(val_dataset)))
    logger.info("len(train_dataset)): {}".format(len(train_dataset)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), 
        shuffle=not args.distributed,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), 
        shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    # Actuall Training
    train_multi_label_coco(model, ema, train_loader, val_loader, args.lr, args, logger)


def train_multi_label_coco(model, ema, train_loader, val_loader, lr, args, logger):
    # set optimizer
    Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, 
        disable_torch_grad_focal_loss=args.dtgfl)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        for i, (inputData, target) in enumerate(train_loader):
            # break
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        model.eval()
        mAP_score_regular, mAP_score_ema = validate_multi(val_loader, model, ema, logger, args)
        model.train()

        if dist.get_rank() == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'mAP': mAP_score_regular
            }, 
            savedir = args.output, 
            savedname = 'model-{}-{}.ckpt'.format(epoch + 1, i + 1), 
            is_best = mAP_score_regular > mAP_score_ema and mAP_score_regular > highest_mAP )

            save_checkpoint({
                'state_dict': ema.module.state_dict(),
                'epoch': epoch,
                'mAP': mAP_score_ema
            }, 
            savedir = args.output, 
            savedname = 'model-{}-{}-{}.ckpt'.format('ema', epoch + 1, i + 1),  
            is_best = mAP_score_ema > mAP_score_regular and mAP_score_ema > highest_mAP)

        mAP_score = max(mAP_score_regular, mAP_score_ema)
        highest_mAP = max(highest_mAP, mAP_score)
        logger.info('current_mAP = {:.4f}, highest_mAP = {:.4f}\n'.format(mAP_score, highest_mAP))


def save_checkpoint(state_dict, savedir, savedname, is_best, rank=0):
    torch.save(state_dict, os.path.join(savedir, savedname))
    if is_best:
        torch.save(state_dict, os.path.join(savedir, 'model-highest.ckpt'))


def validate_multi(val_loader, model, ema_model, logger, args):
    logger.info("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    
    if dist.get_rank() == 0:
        batchs = tqdm(val_loader)
    else:
        batchs = val_loader

    for i, (input, target) in enumerate(batchs):
        # target = target
        target = target.max(dim=1)[0]
        # import ipdb; ipdb.set_trace()
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.detach().cpu())
        preds_ema.append(output_ema.detach().cpu())
        targets.append(target.detach().cpu())

    # saved data
    targets = torch.cat(targets).numpy()
    preds_regular = torch.cat(preds_regular).numpy()
    preds_ema = torch.cat(preds_ema).numpy()

    data_regular = np.concatenate((preds_regular, targets), axis=1)
    saved_name_regular = 'tmpdata/data_regular_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name_regular), data_regular)
    data_ema = np.concatenate((preds_ema, targets), axis=1)
    saved_name_ema = 'tmpdata/data_ema_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name_ema), data_ema)
    if dist.get_world_size() > 1:
        dist.barrier()

    if dist.get_rank() == 0:
        logger.info("Calculating mAP:")
        filenamelist_regular = ['tmpdata/data_regular_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        mAP_score_regular = sl_mAP([os.path.join(args.output, _filename) for _filename in filenamelist_regular], args.num_classes) * 100

        filenamelist_ema = ['tmpdata/data_ema_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        mAP_score_ema = sl_mAP([os.path.join(args.output, _filename) for _filename in filenamelist_ema], args.num_classes) * 100

        logger.info("mAP score regular {:.4f}, mAP score EMA {:.4f}".format(mAP_score_regular, mAP_score_ema))
    else:
        mAP_score_regular = 0
        mAP_score_ema = 0

    # mAP_score_regular = mAP(targets, preds_regular)
    # mAP_score_ema = mAP(targets, preds_ema)
    
    return mAP_score_regular, mAP_score_ema


if __name__ == '__main__':
    main()
