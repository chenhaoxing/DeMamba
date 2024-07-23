import random
import argparse
import yaml
import torch
import os
import util
from util import build_model, train_one_epoch
from dataloader import generate_dataset_loader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of detector in yaml format')
    args = parser.parse_args()

    return args


if __name__ == '__main__': 
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    print("******* Building models. *******")
    print(cfg)
    model = util.build_model(cfg['model'])
    model = model.cuda()

    if cfg['tuning_mode'] == 'lp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
    loss = nn.BCEWithLogitsLoss()
    
    trMaxEpoch = cfg['max_epoch']
    snapshot_path = cfg['save_dir']
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    max_epoch, max_acc = 0, 0

    for epochID in range(0, trMaxEpoch):
        print("******* Training epoch", str(epochID)," *******")
        print("******* Building datasets. *******")
        train_loader, val_loader = generate_dataset_loader(cfg)
        max_epoch, max_acc, epoch_time = train_one_epoch(cfg, model, loss, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path)
        print("******* Ending epoch", str(epochID)," Time ", str(epoch_time), "*******")

