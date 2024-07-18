from torchvision.models import DenseNet
from torchvision.models.densenet import _DenseBlock, _Transition
from .csra import CSRA, MHA
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class DenseNet_CSRA(DenseNet):
    arch_settings = {
        121: (32, (6, 12, 24, 16)),
        169: (32, (6, 12, 32, 32)),
        201: (32, (6, 12, 48, 32)),
        161: (48, (6, 12, 36, 24))
    }

    def __init__(self, num_heads, lam, num_classes, depth=121, growth_rate=32, input_dim=2208, cutmix=None):
        self.growth_rate, self.block_config = self.arch_settings[depth]
        self.depth = depth
        super(DenseNet_CSRA, self).__init__(growth_rate=self.growth_rate, block_config=self.block_config, num_init_features=64)
        self.init_weights(pretrained=True, cutmix=cutmix)

        self.classifier = MHA(num_heads, lam, input_dim, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits

    def backbone(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        return out

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["densenet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the DenseNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        # remove the original classifier
        self.classifier = nn.Sequential()