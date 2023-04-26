# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag
# - Add testing on other datasets, model parts, and dataset splits
# - Save evaluation results to json file

import argparse
import json
import os
from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    cfg.data.val.pipeline[1]['img_scale'] = tuple(
        cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=[
            'same',
            'whole',
            'slide',
        ],
        default='same',
        help='Inference mode.')
    parser.add_argument('--dataset', default='Config')
    parser.add_argument(
        '--model',
        choices=[
            'model',
            'ema_model',
        ],
        default='model',
        help='Submodel to evaluate.')
    parser.add_argument(
        '--train-set',
        action='store_true',
        help='Run inference on the train set')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    assert not (args.train_set and args.test_set)
    eval_set = 'val'
    if args.train_set:
        eval_set = 'train'
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'train')
    if args.test_set:
        eval_set = 'test'
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    if args.dataset == 'Config':
        dataset = build_dataset(cfg.data.test)
    elif args.dataset == 'Cityscapes':
        pipeline = deepcopy(cfg.data.test.pipeline)
        dataset = build_dataset(
            dict(
                type='CityscapesDataset',
                data_root='data/cityscapes/',
                img_dir='leftImg8bit/val',
                ann_dir='gtFine/val',
                pipeline=pipeline))
    elif args.dataset == 'BDD100K':
        pipeline = deepcopy(cfg.data.test.pipeline)
        dataset = build_dataset(
            dict(
                type='BDD100KDataset',
                data_root='data/bdd100k',
                img_dir='images/10k/val',
                ann_dir='labels/sem_seg/masks/val',
                pipeline=pipeline))
    elif args.dataset == 'Mapillary':
        pipeline = deepcopy(cfg.data.test.pipeline)
        dataset = build_dataset(
            dict(
                type='MapillaryDataset',
                data_root='data/mapillary',
                img_dir='validation/images',
                ann_dir='validation/labels',
                pipeline=pipeline))
    else:
        raise NotImplementedError(args.dataset)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), (f'{args.model}.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            res = dataset.evaluate(outputs, args.eval, **kwargs)
            if args.dataset == 'Config':
                res_file = args.checkpoint.replace(
                    '.pth', f'_{args.model}_{eval_set}_iou.json')
            else:
                res_file = args.checkpoint.replace(
                    '.pth', f'_{args.model}_{args.dataset}_'
                    f'{eval_set}_iou.json')
            assert res_file != args.checkpoint
            with open(res_file, 'w') as fp:
                json.dump(res, fp, indent=4)
            print([k for k, v in res.items() if 'IoU' in k])
            print([round(v * 100, 1) for k, v in res.items() if 'IoU' in k])


if __name__ == '__main__':
    main()
