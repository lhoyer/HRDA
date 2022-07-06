import mmcv
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from matplotlib import pyplot as plt
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def visualize(data, result, palette, class_names, out_file, opacity=1.):
    img_meta = data['img_metas'][0].data[0][0]
    h, w, _ = img_meta['img_shape']
    img = tensor2imgs(data['img'][0], **img_meta['img_norm_cfg'])[0]

    label_map = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    for label, (color, name) in enumerate(zip(palette, class_names)):
        if label in result[0]:
            label_map[result[0] == label, :] = color
            plt.plot([], [], color=np.array(color) / 255, label=name)

    if opacity < 1:
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        label_map = img_show * (1 - opacity) + label_map * opacity

    label_map = label_map.astype(np.uint8)

    plt.axis('off')
    plt.legend(loc='lower right', framealpha=.5)
    plt.tight_layout()
    plt.imshow(label_map)
    plt.savefig(out_file)
    plt.clf()

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
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


def def_load_input_data(cfg, img_path):
    # Does not work with creating the pipeline locally=<
    # pipeline = Compose(cfg.data.test['pipeline'])

    dataset = build_dataset(cfg.data.test)
    data = dataset.pipeline({'img_info': {'filename': img_path}, 'img_prefix': ''})

    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    dummy_data = next(iter(data_loader))
    dummy_data['img'][0] = data['img'][0].unsqueeze(0)
    dummy_data['img_metas'][0].data[0][0]['ori_shape'] = data['img_metas'][0].data['ori_shape']
    return dummy_data


class Predictor(BasePredictor):
    def setup(self):
        cfg_path = 'work_dirs/gtaHR2csHR_hrda_246ef/gtaHR2csHR_hrda_246ef.json'
        ckpt_path = 'work_dirs/gtaHR2csHR_hrda_246ef/iter_40000_relevant.pth'

        cfg = mmcv.Config.fromfile(cfg_path)
        cfg = update_legacy_cfg(cfg)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        cfg.model.train_cfg = None

        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        self.checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu',
                                          revise_keys=[(r'^module\.', ''), ('model.', '')])
        self.cfg = cfg
        self.model = MMDataParallel(model, device_ids=[0])

    def predict(self,
                input_image: Path = Input(description="Image of a street view"),
                opacity: float = Input(
                    description="Opacity of the output label-image overlayed over the input image (Float in [0,1], 1 is labels only)",
                    default=0.75),
                ) -> Path:
        input_data = def_load_input_data(self.cfg, str(input_image))
        with torch.no_grad():
            result = self.model(return_loss=False, **input_data)

        out_path = "output.png"
        visualize(input_data, result, self.checkpoint['meta']['PALETTE'], self.checkpoint['meta']['CLASSES'], out_path,
                  float(opacity))

        return Path(out_path)
