"""
Train a model on the Hateful Memes Dataset
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from json import dumps


def main(args):
    # set up logger and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training = True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()

    # dump the args info
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # set seed
    log.info(f'Seed: {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get Model
    log.info("Making model....")
    if(args.model_type == "baseline"):
        model = baseline_model()
    else:
        raise Exception("Model provided not valid")

    model = nn.DataParallel(model, args.gpu_ids)

    # load the step if restarting
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0    

    # send model to dev and start training
    model = model.to(device)
    model.train()
    ema = util.EMA

    # Get checkpoint saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints = args.max_checkpoints,
                                 metric_name = args.metric_name,
                                 maximize_metric = args.maximize_metric,
                                 log = log)
    optimizer = optim.Adam(model.parameters(),
                            lr = args.lr,
                            betas = (0.8, 0.999),
                            eps = 1e-7,
                            weight_decay = args.l2_wd)

    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # load in data
    log.info("Building dataset....")
    train_dataset = HatefulMemes()

if __name__ == '__main__':
    main(get_train_args())