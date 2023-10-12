class Dropout2d(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return F.dropout2d(x, p=self.p, training=True, inplace=self.inplace)

import segmentation_models_pytorch as smp
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# モデル定義
model = smp.DeepLabV3Plus(
    encoder_name='resnet18',
    encoder_weights='imagenet',
    in_channels= 3,
    classes=1,
).to(device)

encoder = model.encoder
encoder.layer3 = torch.nn.Sequential(
    encoder.layer3[0],
    Dropout2d(p=0.5),
    encoder.layer3[1],
    Dropout2d(p=0.5))
model.encoder = encoder # 新たなモデルのencoder部分を置き換える