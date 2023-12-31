import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class Dropout2d(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return F.dropout2d(x, p=self.p, training=True, inplace=self.inplace)
    
class DeepLabv3plusModel:
    def __init__(self):
        self.model = self._create_model()

    def _create_model(self):
        model = smp.DeepLabV3Plus(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )

        encoder = model.encoder
        encoder.layer2 = torch.nn.Sequential(
            encoder.layer2[0],
            Dropout2d(p=0.5),
            encoder.layer2[1],
            Dropout2d(p=0.5)
        )
        encoder.layer3 = torch.nn.Sequential(
            encoder.layer3[0],
            Dropout2d(p=0.5),
            encoder.layer3[1],
            Dropout2d(p=0.5)
        )
        model.encoder = encoder

        return model

    def get_model(self):
        return self.model