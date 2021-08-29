import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer, utilities
from pytorch_lightning.core.lightning import LightningModule

Z_DIM = 100
G_HIDDEN = 64
IMAGE_CHANNEL = 1
D_HIDDEN = 64


class DCGAN(LightningModule):
    def __init__(self):
        super().__init__()

        self.generator = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     x = self.generator(x)
    #     return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        true_label = torch.full((x.size(0),), 1, dtype=torch.float, device=device)
        false_label = torch.full((x.size(0),), 0, dtype=torch.float, device=device)
        if optimizer_idx == 0:
            z_noise = torch.randn(
                x.size(0), Z_DIM, 1, 1, device=device, requires_grad=True
            )
            self.generated_images = self.generator(z_noise)
            d_true_loss = nn.BCELoss()(
                self.discriminator(x).view(-1, 1).squeeze(1), true_label
            )
            d_false_loss = nn.BCELoss()(
                self.discriminator(self.generated_images.detach())
                .view(-1, 1)
                .squeeze(1),
                false_label,
            )
            d_loss = (d_true_loss + d_false_loss) / 2
            self.log("d_loss", d_loss)
            return d_loss

        if optimizer_idx == 1:
            g_loss = nn.BCELoss()(
                self.discriminator(self.generated_images).view(-1, 1).squeeze(1),
                true_label,
            )
            self.log("g_loss", g_loss)
            return g_loss

    def configure_optimizers(self):
        optimizerD = torch.optim.Adam(
            self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        optimizerG = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        return optimizerD, optimizerG

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 50 == 0:
            torchvision.utils.save_image(
                self.generated_images, f"res_{batch_idx}.png", nrow=3, padding=10
            )


if __name__ == "__main__":
    utilities.seed.seed_everything(seed=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = torchvision.datasets.MNIST(
        root="./mnist",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=12
    )

    dcgan = DCGAN()
    trainer = Trainer(gpus=1)
    trainer.fit(dcgan, data_loader)
