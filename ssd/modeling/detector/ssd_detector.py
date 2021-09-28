from torch import nn

from ssd.modeling.backbone import build_backbone
# from ssd.modeling.decoder import build_decoder
from ssd.modeling.fpn import build_fpn
from ssd.modeling.box_head import build_box_head


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.decoder = build_fpn()
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        features = self.decoder(features)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
