# Adapted from: https://github.com/lhoyer/DAFormer

import itertools
import os

from mmcv import Config

# flake8: noqa


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    for j in range(1, 100):
        hrda_name = [e for e in architecture.split('_') if f'hrda{j}' in e]
        for n in hrda_name:
            architecture = architecture.replace(f'{n}_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature, min_crop_ratio):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=min_crop_ratio)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {
            '_base_': ['_base_/default_runtime.py'],
            'gpu_model': gpu_model,
            'n_gpus': n_gpus
        }
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        sync_crop_size_mod = sync_crop_size
        inference_mod = inference
        model_base = get_model_base(architecture_mod, backbone)
        model_base_cfg = Config.fromfile(os.path.join('configs', model_base))
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        hrda_ablation_opts = None
        outer_crop_size = sync_crop_size_mod \
            if sync_crop_size_mod is not None \
            else (int(crop.split('x')[0]), int(crop.split('x')[1]))
        if 'hrda1' in architecture_mod:
            o = [e for e in architecture_mod.split('_') if 'hrda' in e][0]
            hr_crop_size = (int((o.split('-')[1])), int((o.split('-')[1])))
            hr_loss_w = float(o.split('-')[2])
            hrda_ablation_opts = o.split('-')[3:]
            cfg['model']['type'] = 'HRDAEncoderDecoder'
            cfg['model']['scales'] = [1, 0.5]
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['single_scale_head'] = model_base_cfg[
                'model']['decode_head']['type']
            cfg['model']['decode_head']['type'] = 'HRDAHead'
            cfg['model']['hr_crop_size'] = hr_crop_size
            cfg['model']['feature_scale'] = 0.5
            cfg['model']['crop_coord_divisible'] = 8
            cfg['model']['hr_slide_inference'] = True
            cfg['model']['decode_head']['attention_classwise'] = True
            cfg['model']['decode_head']['hr_loss_weight'] = hr_loss_w
            if outer_crop_size == hr_crop_size:
                # If the hr crop is smaller than the lr crop (hr_crop_size <
                # outer_crop_size), there is direct supervision for the lr
                # prediction as it is not fused in the region without hr
                # prediction. Therefore, there is no need for a separate
                # lr_loss.
                cfg['model']['decode_head']['lr_loss_weight'] = hr_loss_w
                # If the hr crop covers the full lr crop region, calculating
                # the FD loss on both scales stabilizes the training for
                # difficult classes.
                cfg['model']['feature_scale'] = 'all' if '_fd' in uda else 0.5

        # HRDA Ablations
        if hrda_ablation_opts is not None:
            for o in hrda_ablation_opts:
                if o == 'fixedatt':
                    # Average the predictions from both scales instead of
                    # learning a scale attention.
                    cfg['model']['decode_head']['fixed_attention'] = 0.5
                elif o == 'nooverlap':
                    # Don't use overlapping slide inference for the hr
                    # prediction.
                    cfg['model']['hr_slide_overlapping'] = False
                elif o == 'singleatt':
                    # Use the same scale attention for all class channels.
                    cfg['model']['decode_head']['attention_classwise'] = False
                elif o == 'blurhr':
                    # Use an upsampled lr crop (blurred) for the hr crop
                    cfg['model']['blur_hr_crop'] = True
                elif o == 'samescale':
                    # Use the same scale/resolution for both crops.
                    cfg['model']['scales'] = [1, 1]
                    cfg['model']['feature_scale'] = 1
                elif o[:2] == 'sc':
                    cfg['model']['scales'] = [1, float(o[2:])]
                    if not isinstance(cfg['model']['feature_scale'], str):
                        cfg['model']['feature_scale'] = float(o[2:])
                else:
                    raise NotImplementedError(o)

        # Setup inference mode
        if inference_mod == 'whole' or crop == '2048x1024':
            assert model_base_cfg['model']['test_cfg']['mode'] == 'whole'
        elif inference_mod == 'slide':
            cfg['model'].setdefault('test_cfg', {})
            cfg['model']['test_cfg']['mode'] = 'slide'
            cfg['model']['test_cfg']['batched_slide'] = True
            crsize = sync_crop_size_mod if sync_crop_size_mod is not None \
                else [int(e) for e in crop.split('x')]
            cfg['model']['test_cfg']['stride'] = [e // 2 for e in crsize]
            cfg['model']['test_cfg']['crop_size'] = crsize
            architecture_mod += '_sl'
        else:
            raise NotImplementedError(inference_mod)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        # DAFormer legacy cropping that only works properly if the training
        # crop has the height of the (resized) target image.
        if 'dacs' in uda and plcrop in [True, 'v1']:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        if 'dacs' in uda and plcrop == 'v2':
            cfg['data']['train'].setdefault('target', {})
            cfg['data']['train']['target']['crop_pseudo_margins'] = \
                [30, 240, 30, 30]
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T, rcs_min_crop)
        if 'dacs' in uda and sync_crop_size_mod is not None:
            cfg.setdefault('data', {}).setdefault('train', {})
            cfg['data']['train']['sync_crop_size'] = sync_crop_size_mod

        # Setup optimizer and schedule
        if 'dacs' in uda or 'minent' in uda or 'advseg' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU')

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
            if rcs_min_crop != 0.5:
                uda_mod += f'-{rcs_min_crop}'
        if 'dacs' in uda and sync_crop_size_mod is not None:
            uda_mod += f'_sf{sync_crop_size_mod[0]}x{sync_crop_size_mod[1]}'
        if 'dacs' in uda:
            if not plcrop:
                pass
            elif plcrop in [True, 'v1']:
                uda_mod += '_cpl'
            elif plcrop[0] == 'v':
                uda_mod += f'_cpl{plcrop[1:]}'
            else:
                raise NotImplementedError(plcrop)
        crop_name = f'_{crop}' if crop != '512x512' else ''
        cfg['name'] = f'{source}2{target}{crop_name}_{uda_mod}_' \
                      f'{architecture_mod}_{backbone}_{schedule}'
        if opt != 'adamw':
            cfg['name'] += f'_{opt}'
        if lr != 0.00006:
            cfg['name'] += f'_{lr}'
        if not pmult:
            cfg['name'] += f'_pm{pmult}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}{crop_name}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    gpu_model = 'NVIDIAGeForceRTX2080Ti'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 1
    rcs_T = None
    rcs_min_crop = 0.5
    plcrop = False
    inference = 'whole'
    sync_crop_size = None
    # -------------------------------------------------------------------------
    # Final HRDA (Table 1)
    # -------------------------------------------------------------------------
    # yapf: disable
    if id == 40:
        seeds = [0, 1, 2]
        # rcs_min_crop=2.0 for HRDA instead of rcs_min_crop=0.5 for
        # DAFormer as HRDA is trained with twice the input resolution, which
        # means that the inputs have 4 times more pixels.
        #         source,      target,         crop,        rcs_min_crop
        gta2cs = ('gtaHR',     'cityscapesHR', '1024x1024', 0.5 * (2 ** 2))
        syn2cs = ('synthiaHR', 'cityscapesHR', '1024x1024', 0.5 * (2 ** 2))
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        # dec, backbone = 'dlv2red', 'r101v1c'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, 'v2'
        inference = 'slide'
        for dataset, architecture, sync_crop_size in [
            (gta2cs, f'hrda1-512-0.1_{dec}', None),
            (syn2cs, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                source, target, crop, rcs_min_crop = dataset
                gpu_model = 'NVIDIATITANRTX'
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # HRDA with Further UDA Methods (Table 2)
    # -------------------------------------------------------------------------
    elif id == 41:
        seeds = [0, 1, 2]
        #        opt,     lr,      schedule,     pmult
        sgd   = ('sgd',   0.0025,  'poly10warm', False)
        adamw = ('adamw', 0.00006, 'poly10warm', True)
        hrda = 'hrda1-512-0.1'
        for source,         target,         crop,           architecture,       backbone,   uda,     opt_hp in [
            ('gtaCAug',     'cityscapes',   '512x512',      'dlv2red',          'r101v1c', 'advseg', sgd),
            ('gtaCAugHR',   'cityscapesHR', '1024x1024',    f'{hrda}_dlv2red',  'r101v1c', 'advseg', sgd),
            ('gtaCAug',     'cityscapes',   '512x512',      'dlv2red',          'r101v1c', 'minent', sgd),
            ('gtaCAugHR',   'cityscapesHR', '1024x1024',    f'{hrda}_dlv2red',  'r101v1c', 'minent', sgd),
            ('gta',         'cityscapes',   '512x512',      'dlv2red',          'r101v1c', 'dacs',   adamw),
            ('gtaHR',       'cityscapesHR', '1024x1024',    f'{hrda}_dlv2red',  'r101v1c', 'dacs',   adamw),
            ('gta',         'cityscapes',   '512x512',      'dlv2red',          'r101v1c', 'daformer_uda',   adamw),
            ('gtaHR',       'cityscapesHR', '1024x1024',    f'{hrda}_dlv2red',  'r101v1c', 'daformer_uda',   adamw),
        ]:
            if uda == 'daformer_uda':
                uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, 'v2'
                if crop == '1024x1024':
                    rcs_min_crop = 0.5 * (2 ** 2)
            for seed in seeds:
                opt, lr, schedule, pmult = opt_hp
                gpu_model = 'NVIDIATITANRTX' if 'HR' in source \
                    else 'NVIDIAGeForceRTX2080Ti'
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Crop Size LR/HR Study for UDA (Figure 4 Left)
    # -------------------------------------------------------------------------
    elif id == 42:
        seeds = [0, 1, 2]
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, 'v2'
        inference = 'slide'
        datasets = [
            # source, target,         crop,       rcs_min_crop,    sync_crop_size
            ('gta',   'cityscapes',   '512x512',  0.5,             (128, 128)),
            ('gta',   'cityscapes',   '512x512',  0.5,             (256, 256)),
            ('gta',   'cityscapes',   '512x512',  0.5,             (384, 384)),
            ('gta',   'cityscapes',   '512x512',  0.5,             None),
            ('gtaHR', 'cityscapesHR', '1024x1024', 0.5 * (2 ** 2), (256, 256)),
            ('gtaHR', 'cityscapesHR', '1024x1024', 0.5 * (2 ** 2), (512, 512)),
            ('gtaHR', 'cityscapesHR', '1024x1024', 0.5 * (2 ** 2), (768, 768)),
        ]
        for (source, target, crop, rcs_min_crop, sync_crop_size), seed in \
                itertools.product(datasets, seeds):
            s = sync_crop_size[0] if sync_crop_size is not None \
                else int(crop.split('x')[0])
            gpu_model = 'NVIDIATITANRTX' if s >= 768 \
                else 'NVIDIAGeForceRTX2080Ti'
            cfg = config_from_vars()
            if crop == '1024x1024' and sync_crop_size == (256, 256):
                # Too memory intensive for small GPU during evaluation
                cfg['model']['test_cfg']['batched_slide'] = False
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Crop Size LR/HR Study for Oracle (Figure 4 Right)
    # -------------------------------------------------------------------------
    elif id == 43:
        seeds = [0, 1, 2]
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda = 'target-only'
        inference = 'slide'
        datasets = [
            # source, target,         crop
            ('',      'cityscapes',   '128x128'),
            ('',      'cityscapes',   '256x256'),
            ('',      'cityscapes',   '384x384'),
            ('',      'cityscapes',   '512x512'),
            ('',      'cityscapesHR', '256x256'),
            ('',      'cityscapesHR', '512x512'),
            ('',      'cityscapesHR', '768x768'),
        ]
        for (source, target, crop), seed in \
                itertools.product(datasets, seeds):
            s = int(crop.split('x')[0])
            gpu_model = 'NVIDIATITANRTX' if s >= 768 \
                else 'NVIDIAGeForceRTX2080Ti'
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # HRDA Crop and Component Ablations (Table 3 - 7)
    # -------------------------------------------------------------------------
    elif id == 44:
        seeds = [0, 1, 2]
        gta2cs = ('gtaHR', 'cityscapesHR', '1024x1024', 0.5 * (2 ** 2))
        gta2csLR = ('gta', 'cityscapes', '512x512', 0.5)
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, 'v2'
        inference = 'slide'
        for dataset, architecture, sync_crop_size in [
            # --------------------------
            # Table 3: Context Crop Size
            # --------------------------
            # (gta2csLR, dec,                    (256, 256)),  # LR0.5; already run in exp 42
            # (gta2cs,   dec,                    (512, 512)),  # HR0.5; already run in exp 42
            (gta2cs,     f'hrda1-512-0.1_{dec}', (512, 512)),  # LR0.5+HR0.5
            (gta2cs,     f'hrda1-512-0.1_{dec}', (768, 768)),  # LR0.75+HR0.5
            # (gta2cs,   f'hrda1-512-0.1_{dec}', None),        # LR1.0+HR0.5; already run in exp 40
            # -------------------------
            # Table 4: Detail Crop Size
            # -------------------------
            # (gta2csLR, dec,                    None),        # LR1.0; already run in exp 42
            # (gta2cs,   dec,                    (256, 256)),  # HR0.25; already run in exp 42
            (gta2cs,     f'hrda1-256-0.1_{dec}', None),        # LR1.0+HR0.25
            (gta2cs,     f'hrda1-384-0.1_{dec}', None),        # LR1.0+HR0.375
            # (gta2cs,   f'hrda1-512-0.1_{dec}', None),        # LR1.0+HR0.5; already run in exp 40
            # ---------------------------------
            # Table 5: Comparison with Naive HR
            # ---------------------------------
            # (gta2cs,   dec,                    (768, 768)),  # HR0.75; already run in exp 42
            (gta2cs,     f'hrda1-384-0.1_{dec}', (768, 768)),  # LR0.75+HR0.375
            # (gta2cs,   f'hrda1-512-0.1_{dec}', None),        # LR1.0+HR0.5; already run in exp 40
            # -----------------------------
            # Table 6: Detail Crop Variants
            # -----------------------------
            # (gta2csLR, dec,                              None),  # LR1.0; already run in exp 42
            (gta2csLR,   f'hrda1-256-0.1-samescale_{dec}', None),  # LR1.0+LR0.5
            (gta2cs,     f'hrda1-512-0.1-blurhr_{dec}',    None),  # LR1.0+Up-LR0.5
            # (gta2cs,   f'hrda1-512-0.1_{dec}',           None),  # already run in exp 40
            # ----------------------------
            # Table 7: Component Ablations
            # ----------------------------
            # (gta2cs,   dec,                                     (512, 512)),  # Row 1; already run in exp 42
            # (gta2csLR, dec,                                     None),        # Row 2; already run in exp 42
            (gta2cs,     f'hrda1-512-0-fixedatt-nooverlap_{dec}', None),        # Row 3
            (gta2cs,     f'hrda1-512-0-nooverlap_{dec}',          None),        # Row 4
            (gta2cs,     f'hrda1-512-0_{dec}',                    None),        # Row 5
            # (gta2cs,   f'hrda1-512-0.1_{dec}',                  None),        # Row 6; already run in exp 40
            # -----------------------------
            # Figure S1: Detail Loss Weight
            # -----------------------------
            # (gta2cs, f'hrda1-512-0_{dec}',   None),  # already run above for Tab. 7
            # (gta2cs, f'hrda1-512-0.1_{dec}', None),  # already run in exp 40
            (gta2cs,   f'hrda1-512-0.2_{dec}', None),
            (gta2cs,   f'hrda1-512-0.3_{dec}', None),
            (gta2cs,   f'hrda1-512-0.4_{dec}', None),
            (gta2cs,   f'hrda1-512-0.5_{dec}', None),
            (gta2cs,   f'hrda1-512-0.6_{dec}', None),
            (gta2cs,   f'hrda1-512-0.7_{dec}', None),
            # ---------------------------------
            # Table S1: Context Crop Resolution
            # ---------------------------------
            # (gta2cs, dec,                           (512, 512)),  # No context crop; already run in exp42
            (gta2cs,   f'hrda1-512-0.1-sc0.25_{dec}', (512, 512)),  # s_c=4
            # (gta2cs, f'hrda1-512-0.1_{dec}',        (512, 512)),  # s_c=2; already run above for Tab. 3
            (gta2cs,   f'hrda1-512-0.1-sc0.75_{dec}', (512, 512)),  # s_c=1.33
        ]:
            for seed in seeds:
                source, target, crop, rcs_min_crop = dataset
                gpu_model = 'NVIDIATITANRTX'
                cfg = config_from_vars()
                cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
