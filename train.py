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
    torch.manual_seed(args.seed)
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
    train_dataset = HatefulMemes(args.train_eval_file,
                                 args.img_folder_rel_path,
                                 args.text_model_path)
    ###############
    ################## might need to modify collate fn to allow for padding of text data
    ###############
    train_loader = data.DataLoader(train_dataset,
                                   batch_size = args.batch_size,
                                   shuffle = True,
                                   num_workers = args.num_workers)
    dev_dataset = HatefulMemes(args.dev_eval_file,
                                args.img_folder_rel_path,
                                args.text_model_path)                             
    dev_loader = data.DataLoader(dev_dataset,
                                   batch_size = args.batch_size,
                                   shuffle = True,
                                   num_workers = args.num_workers)
    # Start training
    log.info("Training...")
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}....')
        with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
            for img_id, image, text, label in train_loader:
                # forward pass here
                image = image.to(device)
                text = text.to(device)

                batch_size = args.batch_size
                optimizer.zero_grad()

                if(args.model_type == "baseline"):
                    log_softmax_score = model(image, text)
                else:
                    raise Exception("Model Type Invalid")

                # calc loss
                label = label.to(device)
                loss = F.nll_loss(log_softmax_score, label)
                loss_val = loss.item()

                # backward pass here
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step//batch_size)

                # log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch = epoch,
                                         loss = loss_val)
                tbx.add_scalar("train/loss", loss_val, step)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]['lr'],
                               step)
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Eval and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    results, pred_dict = evaluate(args, model, dev_loader, device)
                    saver.save(step, model, results[args.metric_name], device)

                    results_str = ", ".join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # tensorboard
                    log.info("Visualizing in TensorBoard")
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    
def evaluate(args, model, data_loader, device):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {} # id, prob and prediction


    acc = 0
    num_correct, num_samples = 0, 0
    
    with torch.no_grad(), \
        tqdm(total=len(data_loader.dataset)) as progress_bar:
        for img_id, image, text, label in data_loader:
            # forward pass here
            image = image.to(device)
            text = text.to(device)

            batch_size = args.batch_size
            optimizer.zero_grad()

            if(args.model_type == "baseline"):
                log_softmax_score = model(image, text)
            else:
                raise Exception("Model Type Invalid")

            # calc loss
            label = label.to(device)
            loss = F.nll_loss(log_softmax_score, label)
            nll_meter.update(loss.item(), batch_size)

            # get acc and auroc
            softmax_score = log_softmax_score.exp()
            _, preds = softmax_score.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)

            pred_dict_update = util.make_update_dict(
                img_id,
                preds,
                softmax_score,
                label
            )

            # update 
            pred_dict.update(pred_dict_update)

        acc = float(num_correct) / num_samples
    model.train()

    results_list = [("Loss", nll_meter.avg)
                    ("Acc", acc)]
    results = OrderedDict(results_list)
    
    return results, pred_dict




    



    
if __name__ == '__main__':
    main(get_train_args())