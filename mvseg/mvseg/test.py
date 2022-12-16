from calib.calib.utils.experiments import load_experiment
import glob
import torch
from calib.calib.datasets import get_dataset

import argparse
from pathlib import Path
import signal
import shutil
import re
import os
import copy
from collections import defaultdict

from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import get_dataset
from .models import get_model
from .utils.stdout_capturing import capture_outputs
from .utils.tools import AverageMetric, MedianMetric, set_seed, fork_rng
from .utils.tensor import batch_to_device
from .utils.experiments import (
    delete_old_checkpoints, get_last_checkpoint, get_best_checkpoint)
from ..settings import TRAINING_PATH
from .. import logger

default_test_conf = {
        }  # will overwrite the training and default configurations

def do_test(model, loader, device, loss_fn, metrics_fn, conf, pbar=True):
    
    model = model.eval()  # optionally move the model to GPU
    results = {}
    
    for data in tqdm(loader, desc='Testing', ascii=True, disable=not pbar):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data
        numbers = {**metrics, **{'loss/'+k: torch.tensor([v]) for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
            results[k].update(v)
    results = {k: results[k].compute() for k in results}
    return results

def main(conf, output_dir, args):
    
    writer = SummaryWriter(log_dir=str(output_dir))
    data_conf = copy.deepcopy(conf.data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')
    dataset = get_dataset(data_conf.name)(data_conf)
    test_loader = dataset.get_data_loader('test')
    logger.info(f'Test loader has {len(test_loader)} batches')
    
    model = load_experiment(args.experiment, conf.model)
    if torch.cuda.is_available():
        model.cuda()
    loss_fn, metrics_fn = model.loss, model.metrics

    results = do_test(model, test_loader, device, loss_fn, metrics_fn, conf, pbar=True)
    str_results = [f'{k} {v:.3E}' for k, v in results.items()]
    logger.info(f'[Testing] {{{", ".join(str_results)}}}')
    for k, v in results.items():
        writer.add_scalar('test/'+k, v, 0)
    torch.cuda.empty_cache()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str) # to load model
    parser.add_argument('--conf', type=str)
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    logger.info(f'Starting test {args.experiment}')
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)

    if conf.train.seed is None:
        conf.train.seed = torch.initial_seed() & (2**32 - 1)
    OmegaConf.save(conf, str(output_dir / 'config.yaml'))

    main(conf, output_dir, args)

