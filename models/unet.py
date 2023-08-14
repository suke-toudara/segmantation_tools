
import monai
from monai.networks.blocks import Convolution
from monai.networks.nets import UNet
     
     
class Unet(pl.LightningModule):

    def __init__(self, lr: float):
        super().__init__()

        self.lr = lr

        self.unet = UNet(
                  dimensions=2, in_channels=1, out_channels=1,
                  channels=(64, 128, 256, 512, 1024),
                  strides=(2, 2, 2, 2, 2)
        )


    def forward(self, x):
        h = self.unet(x)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.binary_cross_entropy_with_logits(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy(y.sigmoid(), t.int()), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou(y.sigmoid(), t.int()), on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.binary_cross_entropy_with_logits(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.sigmoid(), t.int()), on_step=False, on_epoch=True)
        self.log('val_iou', iou(y.sigmoid(), t.int()), on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer