import mmcv
import torch
from mmcv.runner.hooks import EvalHook

from mmseg.models.utils.farthest_point_sampling import \
    farthest_point_sample_tensor


class StyleHallucinationHook(EvalHook):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_train_iter(self, runner):
        # Initialize style hallucination at the beginning of training
        if self.initial_flag:
            self._do_evaluate(runner)

        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        # Based on: https://github.com/HeliosZhao/SHADE/blob/ea4214ad4eaa1ba2210656bff315afb6c6f50e28/train.py#L586  # noqa
        if not self.initial_flag and not self._should_evaluate(runner):
            return

        model_training_state = runner.model.training
        runner.model.eval()
        prog_bar = mmcv.ProgressBar(len(self.dataloader))
        style_list = []
        with torch.no_grad():
            for i, data_batch in enumerate(self.dataloader):
                style_feats = runner.model(
                    **data_batch, return_loss=False, return_style=True)
                img_mean = style_feats.mean(dim=[2, 3])  # B,C
                img_sig = style_feats.std(dim=[2, 3])  # B,C
                img_statis = torch.stack((img_mean, img_sig), dim=1)  # B,2,C
                style_list.append(img_statis)
                prog_bar.update()

        style_list = torch.cat(style_list, dim=0)
        base_style_num = style_list.size(-1)
        style_list = style_list.reshape(style_list.size(0), -1).detach()

        proto_styles, centroids = farthest_point_sample_tensor(
            style_list, base_style_num)  # C,2C
        proto_styles = proto_styles.reshape(base_style_num, 2, base_style_num)
        proto_mean, proto_std = proto_styles[:, 0], proto_styles[:, 1]

        runner.model.module.model.backbone.style_hallucination.\
            proto_mean.copy_(proto_mean)
        runner.model.module.model.backbone.style_hallucination.\
            proto_std.copy_(proto_std)

        runner.model.train(model_training_state)
