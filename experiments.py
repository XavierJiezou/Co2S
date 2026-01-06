from argparse import ArgumentParser
from functools import reduce
import itertools
import yaml
import os
import os.path as osp
import subprocess
import collections.abc
from version import __version__


DATA_DIR = 'data'


def nested_set(dic, key, value):
    keys = key.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def nested_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_git_revision() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return ''

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def config_from_vars(
    exp_id,
    gpu_model='3090',
    n_gpus=4,
    n_nodes=1,
    batch_size=1,
    epochs=80,
    iters=None,
    scheduler_max_iters=None,
    dataset='whdld',
    split='1_4',
    img_scale=[2048, 512],
    scale_ratio_range=(0.5, 2.0),
    crop_size=512,
    labeled_photometric_distortion=False,
    renorm_clip_img=False,
    method='Co2S',
    use_fp=True,
    conf_mode='pixelwise',
    conf_thresh=0.95,
    pleval=True,
    disable_dropout=True,
    fp_rate=0.5,
    mcc_text='same',
    pl_text='same',
    opt='adamw',
    lr=1e-4,
    backbone_lr_mult=10.0,
    conv_enc_lr_mult=1.0,
    warmup_iters=0,
    criterion='mmseg',
    criterion_u='mmseg',
    model='mmseg.dual_model_clip',
    text_embedding_variant='single',
    eval_mode='zegclip_sliding_window',
    eval_every=1,
    nccl_p2p_disable=1,
    nccl_debug='INFO',
):
    cfg = dict()
    name = ''

    # Dataset
    cfg['dataset'] = dataset
    name += dataset
    cfg['data_root'] = dict(
        whdld = osp.join(DATA_DIR, 'whdld/'),
        loveda = osp.join(DATA_DIR, 'loveda/'),
        potsdam = osp.join(DATA_DIR, 'potsdam/'),
        gid = osp.join(DATA_DIR, 'gid/'),
        mer = osp.join(DATA_DIR, 'mer/'),
        msl = osp.join(DATA_DIR, 'msl/'),        
    )[dataset]
    cfg['nclass'] = dict(
        whdld=6,
        loveda=7,
        potsdam=6,
        gid=15,
        mer=9,
        msl=9,
    )[dataset]
    
    if cfg['dataset'] not in ['gid','mer','msl']:
        cfg['reduce_zero_label'] = True
        
    cfg['split'] = split
    name += f'-{split}'
    cfg['img_scale'] = img_scale
    
    if img_scale is not None:
        name += f'-{img_scale}'
    cfg['scale_ratio_range'] = scale_ratio_range
    if scale_ratio_range != (0.5, 2.0):
        name += f'-s{scale_ratio_range[0]}-{scale_ratio_range[1]}'
    cfg['crop_size'] = crop_size
    name += f'-{crop_size}'
    cfg['labeled_photometric_distortion'] = labeled_photometric_distortion
    if labeled_photometric_distortion:
        name += '-phd'

    # Model
    name += f'_{model}'.replace('mmseg.', '').replace('zegclip', 'zcl')
    cfg['model_args'] = {}
    if model == 'dlv3p-r101':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'resnet101'
        cfg['replace_stride_with_dilation'] = [False, False, True]
        cfg['dilations'] = [6, 12, 18]
    elif model == 'dlv3p-xc65':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'xception'
        cfg['dilations'] = [6, 12, 18]
    else:
        cfg['model'] = model
        cfg['text_embedding_variant'] = text_embedding_variant
        cfg['mcc_text'] = text_embedding_variant if mcc_text == 'same' else mcc_text
        cfg['pl_text'] = text_embedding_variant if pl_text == 'same' else pl_text
        text_variant_abbrev = {
            'conceptavg_single': 'cavgs',
            'conceptavg2_single': 'cavg2s',
            'conceptavg3_single': 'cavg3s',
            'conceptavg4_single': 'cavg4s',
            'conceptavg5_single': 'cavg5s',
            'conceptavg6_single': 'cavg6s',
            'concept2_single': 'c2s',
            'concept3_single': 'c3s',
            'concept4_single': 'c4s',
            'concept5_single': 'c5s',
            'concept6_single': 'c6s',
            'multi': 'm',
        }
        if text_embedding_variant != 'single':
            name += '-t' + text_variant_abbrev[text_embedding_variant]
        if mcc_text != 'same':
            name += '-mt' + text_variant_abbrev[mcc_text]
        if pl_text != 'same':
            name += '-pt' + text_variant_abbrev[pl_text]


    # Method
    cfg['method'] = method
    name += f'_{method}'
    
    cfg['use_fp'] = use_fp
    if not use_fp:
        name += '-nfp'
    cfg['conf_mode'] = conf_mode
    name += {
        'pixelwise': '',
        'pixelratio': '-cpr',
        'pixelavg': '-cpa',
    }[conf_mode]
    cfg['conf_thresh'] = conf_thresh
    name += f'-{conf_thresh}'
        
    cfg['disable_dropout'] = disable_dropout
    if disable_dropout:
        name += '-disdrop'
    
    cfg['pleval'] = pleval
    if pleval:
        name += '-plev'
            
    cfg['fp_rate'] = fp_rate
    if fp_rate != 0.5:
        name += f'-fpr{fp_rate}'

    if renorm_clip_img:
        cfg['model_args']['renorm_clip_img'] = True
        name += '-rnci'

    # Criterion
    cfg['criterion'] = dict(
        name=criterion,
        kwargs=dict(ignore_index=255)
    )
    if cfg['criterion'] == 'OHEM':
        cfg['criterion']['kwargs'].update(dict(
            thresh=0.7,
            min_kept=200000
        ))
    if criterion != 'mmseg':
        name += f'-{criterion}'.replace('CELoss', 'ce').replace('OHEM', 'oh')
    cfg['criterion_u'] = criterion_u
    if criterion_u != 'mmseg':
        name += f'-u{criterion_u}'.replace('CELoss', 'ce')

    # Optimizer
    if opt == 'original':
        cfg['lr'] = lr
        cfg['lr_multi'] = 10.0 
    elif opt == 'adamw':
        cfg['optimizer'] = dict(
            type='AdamW', lr=lr, weight_decay=0.01,
            paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=backbone_lr_mult),
                                            'text_encoder': dict(lr_mult=0.0),
                                            'conv_encoder': dict(lr_mult=conv_enc_lr_mult),
                                            'norm': dict(decay_mult=0.),
                                            'ln': dict(decay_mult=0.),
                                            'head': dict(lr_mult=10.),
                                            }))
    else:
        raise NotImplementedError(opt)
    name += f'_{opt}-{lr:.0e}'.replace('original', 'org')
    if backbone_lr_mult != 10.0:
        name += f'-b{backbone_lr_mult}'
    if conv_enc_lr_mult != 1.0:
        name += f'-cl{conv_enc_lr_mult}'
    cfg['warmup_iters'] = warmup_iters
    cfg['warmup_ratio'] = 1e-6
    if warmup_iters > 0:
        name += f'-w{human_format(warmup_iters)}'

    # Batch
    cfg['gpu_model'] = gpu_model
    cfg['n_gpus'] = n_gpus
    cfg['n_nodes'] = n_nodes
    cfg['batch_size'] = batch_size
    if n_gpus != 4 or batch_size != 2 or n_nodes != 1:
        name += f'_{n_nodes}x{n_gpus}x{batch_size}'

    # Schedule
    assert not (iters is not None and epochs is not None)
    cfg['epochs'] = epochs
    cfg['iters'] = iters
    if epochs is not None and epochs != 80:
        name += f'-ep{human_format(epochs)}'
    if iters is not None:
        name += f'-i{human_format(iters)}'
    if scheduler_max_iters is not None:
        cfg['scheduler_max_iters'] = scheduler_max_iters
        name += f'-smi{scheduler_max_iters}'

    # Eval
    cfg['eval_mode'] = eval_mode
    if cfg.get('eval_mode') == 'zegclip_sliding_window':
        crop = cfg.get('crop_size', 512)
        overlap_ratio = 1/6
        cfg['stride'] = int(crop * (1 - overlap_ratio))
        
    name += '_e' + {
        'original': 'or',
        'sliding_window': 'sw',
        'zegclip_sliding_window': 'zsw',
    }[eval_mode]
    cfg['eval_every_n_epochs'] = eval_every
    cfg['nccl_p2p_disable'] = nccl_p2p_disable
    cfg['nccl_debug'] = nccl_debug


    cfg['exp'] = exp_id
    cfg['name'] = name.replace('.0_', '').replace('.0-', '').replace('.', '').replace('True', 'T')\
        .replace('False', 'F').replace('None', 'N').replace('[', '')\
        .replace(']', '').replace('(', '').replace(')', '').replace(',', 'j')\
        .replace(' ', '')
    cfg['version'] = __version__
    cfg['git_rev'] = get_git_revision()

    return cfg

def generate_experiment_cfgs(exp_id):
    cfgs = []

    # -------------------------------------------------------------------------
    # Co2S on WHDLD
    # -------------------------------------------------------------------------
    if exp_id == 40:
        n_repeat = 1
        splits = ['1_24', '1_16', '1_8', '1_4']
        list_kwargs = [
            dict(model='mmseg.dual_model_clip', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
        ]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='whdld',
                split=str(split),
                batch_size=2,
                crop_size=256,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Co2S on LoveDA
    # -------------------------------------------------------------------------
    elif exp_id == 41:
        n_repeat = 1
        splits = ['1_40','1_16', '1_8', '1_4']
        list_kwargs = [
            dict(model='mmseg.dual_model_clip', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
        ]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='loveda',
                split=str(split),
                crop_size=512,
                img_scale=None, 
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Co2S on Potsdam
    # -------------------------------------------------------------------------
    elif exp_id == 42:
        n_repeat = 1
        splits = ['1_32','1_16', '1_8', '1_4']
        kwargs_list = [
            dict(model='mmseg.dual_model_clip', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
        ]
        for kwargs, split, _ in itertools.product(kwargs_list, splits, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='potsdam',
                split=str(split),
                crop_size=512,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Co2S on GID-15
    # -------------------------------------------------------------------------
    elif exp_id == 43:
        n_repeat = 1
        splits = ['1_8', '1_4']
        kwargs_list = [
            dict(model='mmseg.dual_model_clip', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
        ]
        for kwargs, split, _ in itertools.product(kwargs_list, splits, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='gid',
                split=str(split),
                crop_size=512,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Co2S on MER
    # -------------------------------------------------------------------------
    elif exp_id == 44:
        n_repeat = 1
        splits = ['1_8', '1_4']
        kwargs_list = [
            dict(model='mmseg.dual_model_clip', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
        ]
        for kwargs, split, _ in itertools.product(kwargs_list, splits, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                batch_size=1,
                epochs=150,
                dataset='mer',
                split=str(split),
                crop_size=512,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Co2S on MSL
    # -------------------------------------------------------------------------
    elif exp_id == 45:
        n_repeat = 1
        splits = ['1_8', '1_4']
        kwargs_list = [
            dict(model='mmseg.dual_model_clip', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
        ]
        for kwargs, split, _ in itertools.product(kwargs_list, splits, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                batch_size=1,
                epochs=80,
                dataset='msl',
                split=str(split),
                crop_size=512,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)    
    else:
        raise NotImplementedError(f'Unknown id {exp_id}')

    return cfgs


def save_experiment_cfgs(exp_id):
    cfgs = generate_experiment_cfgs(exp_id)
    cfg_files = []
    for cfg in cfgs:
        cfg_file = f"configs/generated/exp-{cfg['exp']}/{cfg['name']}.yaml"
        os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
        with open(cfg_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=None, sort_keys=False, indent=2)
        cfg_files.append(cfg_file)

    return cfgs, cfg_files

def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate experiment configs')
    parser.add_argument('--exp', type=int, help='Experiment id')
    parser.add_argument('--run', type=int, default=0, help='Run id')
    parser.add_argument('--ngpus', type=int, default=None, help='Override number of GPUs')
    args = parser.parse_args()

    cfgs, cfg_files = save_experiment_cfgs(args.exp)

    if args.ngpus is None:
        ngpus = cfgs[args.run]["n_gpus"]
    else:
        ngpus = args.ngpus

    cmd = f'bash scripts/train.sh {cfgs[args.run]["method"]} {cfg_files[args.run]} {ngpus}'
    print(cmd)
    run_command(cmd)
