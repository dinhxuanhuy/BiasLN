# Training entry point (DeepfakeBench-derived, extended for the BiasLN revision).
#
# Checkpoint-selection protocols (``--selection``)
#   val          select on ``--val_dataset`` read from ``--val_split`` (default:
#                FaceForensics++ 'val' split). Disjoint from every test set.
#   last         no selection: the last epoch is the model (target-agnostic).
#   test_oracle  LEGACY: select on the *test* split of ``--test_dataset``.
#                Only for reproducing old numbers / measuring the oracle gap.
#
# Every epoch also stores a compact checkpoint (trainable tensors only) in
# <run_dir>/epochs/, so any epoch can be evaluated post hoc (training curves,
# oracle-vs-protocol gap) without re-training.

import argparse
import datetime
import json
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import yaml
from torch.utils.data.distributed import DistributedSampler

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR
from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *  # noqa: F401,F403  (DeepfakeAbstractBaseDataset, samplers)
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from config_utils import apply_overrides, dump_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser(description='Train a detector.')
parser.add_argument('--detector_path', type=str,
                    default='training/config/detector/BiasLN.yaml')
parser.add_argument('--train_dataset', nargs='+')
parser.add_argument('--test_dataset', nargs='+',
                    help='only used by --selection test_oracle (legacy)')
parser.add_argument('--selection', choices=['val', 'last', 'test_oracle'], default=None)
parser.add_argument('--val_dataset', nargs='+', default=None)
parser.add_argument('--val_split', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--rgb_dir', type=str, default=None)
parser.add_argument('--json_dir', type=str, default=None)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--opts', nargs='*', default=[],
                    help='config overrides, e.g. tuning_mode=lora optimizer.adam.lr=1e-4')
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
parser.add_argument('--weights_path', type=str, default=None)
args = parser.parse_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    seed = int(config['manualSeed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def prepare_training_data(config):
    train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')
    use_balanced = bool(config.get('use_class_balanced_sampler', False))
    generator = torch.Generator()
    generator.manual_seed(int(config['manualSeed']))
    if config['ddp']:
        if use_balanced:
            raise ValueError('use_class_balanced_sampler is single-process only.')
        sampler = DistributedSampler(train_set)
        return torch.utils.data.DataLoader(
            dataset=train_set, batch_size=config['train_batchSize'],
            num_workers=int(config['workers']), collate_fn=train_set.collate_fn,
            sampler=sampler, worker_init_fn=_worker_init_fn)
    sampler = (build_class_balanced_sampler(train_set.label_list, seed=config['manualSeed'])
               if use_balanced else None)
    return torch.utils.data.DataLoader(
        dataset=train_set, batch_size=config['train_batchSize'],
        shuffle=(sampler is None), sampler=sampler,
        num_workers=int(config['workers']), collate_fn=train_set.collate_fn,
        worker_init_fn=_worker_init_fn, generator=generator)


def prepare_eval_data(config, names, split):
    """Evaluation loaders in 'test' mode (no augmentation) reading ``split``."""
    loaders = {}
    for name in names:
        cfg = config.copy()
        cfg['test_dataset'] = name
        cfg['data_split'] = split
        eval_set = DeepfakeAbstractBaseDataset(config=cfg, mode='test')
        loaders[name] = torch.utils.data.DataLoader(
            dataset=eval_set, batch_size=config['test_batchSize'], shuffle=False,
            num_workers=int(config['workers']), collate_fn=eval_set.collate_fn,
            drop_last=False)
    return loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    opt_cfg = config['optimizer'][opt_name]
    if hasattr(model, 'get_param_groups'):
        params = model.get_param_groups(opt_cfg['lr'])
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == 'sgd':
        return optim.SGD(params, lr=opt_cfg['lr'], momentum=opt_cfg['momentum'],
                         weight_decay=opt_cfg['weight_decay'])
    if opt_name == 'adam':
        return optim.Adam(params, lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'],
                          betas=(opt_cfg['beta1'], opt_cfg['beta2']), eps=opt_cfg['eps'],
                          amsgrad=opt_cfg['amsgrad'])
    if opt_name == 'adamw':
        return optim.AdamW(params, lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'],
                           betas=(opt_cfg['beta1'], opt_cfg['beta2']), eps=opt_cfg['eps'])
    if opt_name == 'sam':
        return SAM(params, optim.SGD, lr=opt_cfg['lr'], momentum=opt_cfg['momentum'])
    raise NotImplementedError(f'Optimizer {opt_name} is not implemented')


def choose_scheduler(config, optimizer):
    name = config.get('lr_scheduler')
    if name is None:
        return None
    if name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'],
                                         gamma=config['lr_gamma'])
    if name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_T_max'],
                                                    eta_min=config['lr_eta_min'])
    if name == 'linear':
        return LinearDecayLR(optimizer, config['nEpochs'], int(config['nEpochs'] / 4))
    raise NotImplementedError(f'Scheduler {name} is not implemented')


def load_config():
    with open(args.detector_path) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(ROOT, 'training/config/train_config.yaml')) as f:
        base = yaml.safe_load(f)
    # Union of train/test label dictionaries so any dataset can serve as validation.
    with open(os.path.join(ROOT, 'training/config/test_config.yaml')) as f:
        test_labels = yaml.safe_load(f).get('label_dict', {})
    base['label_dict'] = {**test_labels, **base.get('label_dict', {})}
    if 'label_dict' in config:
        base['label_dict'].update(config['label_dict'])
    # Detector YAML wins over the generic train_config for model-specific keys.
    merged = dict(base)
    merged.update({k: v for k, v in config.items() if k not in ('label_dict',)})
    config = merged

    cli = {
        'train_dataset': args.train_dataset, 'test_dataset': args.test_dataset,
        'selection': args.selection, 'val_dataset': args.val_dataset,
        'val_split': args.val_split, 'manualSeed': args.seed, 'rgb_dir': args.rgb_dir,
        'dataset_json_folder': args.json_dir, 'log_dir': args.log_dir,
        'pretrained': args.weights_path,
    }
    config.update({k: v for k, v in cli.items() if v is not None})
    apply_overrides(config, args.opts)
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    config['local_rank'] = args.local_rank
    config['ddp'] = args.ddp
    config.setdefault('selection', 'val')
    config.setdefault('val_dataset', ['FaceForensics++'])
    config.setdefault('val_split', 'val')
    if config.get('dry_run'):
        config['nEpochs'] = 0
        config['save_feat'] = False
    return config


def main():
    config = load_config()
    selection = config['selection']
    if selection == 'val':
        eval_names, eval_split = list(config['val_dataset']), config['val_split']
        overlap = set(eval_names) & set(config.get('test_dataset') or [])
        if overlap and eval_split == 'test':
            raise ValueError(f'validation reads the test split of {overlap}; use '
                             '--selection test_oracle if this is really intended.')
    elif selection == 'test_oracle':
        eval_names, eval_split = list(config['test_dataset']), 'test'
    else:
        eval_names, eval_split = [], None
    config['selection_key'] = (
        None if not eval_names else ('avg' if len(eval_names) > 1 else eval_names[0]))
    config['save_avg'] = len(eval_names) > 1

    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = args.run_name or f"{config['model_name']}_{selection}_s{config['manualSeed']}_{timenow}"
    run_dir = os.path.join(config['log_dir'], run_name)
    config['run_dir'] = run_dir
    os.makedirs(run_dir, exist_ok=True)
    logger = create_logger(os.path.join(run_dir, 'training.log'))
    logger.info(f'Run dir: {run_dir}')
    logger.info(f'Selection protocol: {selection} on {eval_names} (split={eval_split})')
    logger.info('Configuration:\n' + yaml.safe_dump(config, sort_keys=True))
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=True)

    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    train_loader = prepare_training_data(config)
    eval_loaders = prepare_eval_data(config, eval_names, eval_split) if eval_names else None

    model = DETECTOR[config['model_name']](config)
    if config.get('pretrained') and os.path.exists(str(config['pretrained'])):
        ckpt = torch.load(config['pretrained'], map_location='cpu')
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f'Loaded {config["pretrained"]}: missing={len(missing)} unexpected={len(unexpected)}')

    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = config['metric_scoring']
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    summary = getattr(model, 'trainable_param_summary', None)
    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer._bare_model().epoch = epoch
        best_metric = trainer.train_epoch(epoch=epoch, train_data_loader=train_loader,
                                          test_data_loaders=eval_loaders)
        if best_metric is not None:
            logger.info(f'===> Epoch[{epoch}] best so far: {parse_metric_for_print(best_metric)}')
        if config.get('save_every_epoch', True) and config['save_ckpt']:
            trainer.save_epoch_ckpt(epoch)
        if scheduler is not None and config.get('scheduler_step_per_epoch', True):
            scheduler.step()
            logger.info(f"lr after epoch {epoch}: {[g['lr'] for g in optimizer.param_groups]}")

    final_ckpt = None
    if config['save_ckpt'] and config['nEpochs'] > 0:
        last = os.path.join(run_dir, 'epochs', f"ckpt_epoch{config['nEpochs'] - 1:02d}.pth")
        if selection == 'last':
            final_ckpt = last
        elif config['selection_key'] is not None:
            final_ckpt = os.path.join(run_dir, 'test', config['selection_key'], 'ckpt_best.pth')

    info = {
        'run_dir': run_dir,
        'model_name': config['model_name'],
        'seed': config['manualSeed'],
        'selection': selection,
        'selection_datasets': eval_names,
        'selection_split': eval_split,
        'selected': getattr(trainer, 'best_selection', None),
        'final_ckpt': final_ckpt,
        'trainable_params': summary,
        'trainable_breakdown': getattr(model, 'trainable_counts', None),
        'epoch_stats': trainer.epoch_stats,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'amp': config.get('amp', 'none'),
        'batch_size': config['train_batchSize'],
    }
    dump_json(info, os.path.join(run_dir, 'run_info.json'))
    logger.info(f'run_info: {json.dumps(info, default=str)}')
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
