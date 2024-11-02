# AutoEncoder(3,256,256)
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()

        # エンコーダー
        self.encoder = nn.Sequential(
            # 3,32,32→8,16,16
            nn.Conv2d(3, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 8,16,16→16,8,8
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 16,8,8→32,4,4
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        self.fc_mu = nn.Linear(16 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(16 * 4 * 4, latent_dim)

        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, 16 * 4 * 4),
            nn.ReLU(True)
        )

        # デコーダー
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        z = self.reparameterize(mu, logvar)

        x = self.decoder1(z)
        x = x.view(x.size(0), 16, 4, 4)
        x = self.decoder2(x)
        return x, mu, logvar, z


def vae_loss(reconstructed_x, x, mu, logvar):
    # 再構築損失（MSE損失）
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed_x, x, reduction='sum')

    # KLダイバージェンス損失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 二つの損失を組み合わせた損失関数（ELBO）
    elbo = reconstruction_loss + kl_loss

    return elbo