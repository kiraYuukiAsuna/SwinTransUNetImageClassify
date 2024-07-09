import ml_collections
import torch
from torch import nn

import SwinTransformer3D


def get_3D_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True  # skip connection
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.pat_merg_rf = 4
    config.img_size = (128, 128, 128)
    config.reg_head_chan = 16
    return config


class SwinTransformerClassify(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_3D_config()
        self.model = SwinTransformer3D.SwinTransformer3D(
            patch_size=self.config.patch_size,
            in_chans=self.config.in_chans,
            embed_dim=self.config.embed_dim,
            depths=self.config.depths,
            num_heads=self.config.num_heads,
            window_size=self.config.window_size,
            mlp_ratio=self.config.mlp_ratio,
            qkv_bias=self.config.qkv_bias,
            drop_rate=self.config.drop_rate,
            drop_path_rate=self.config.drop_path_rate,
            ape=self.config.ape,
            spe=self.config.spe,
            rpe=self.config.rpe,
            patch_norm=self.config.patch_norm,
            use_checkpoint=self.config.use_checkpoint,
            out_indices=self.config.out_indices,
            pat_merg_rf=self.config.pat_merg_rf,
        )
        self.up0 = SwinTransformer3D.DecoderBlock(self.config.embed_dim * 8, self.config.embed_dim * 4,
                                                  skip_channels=self.config.embed_dim * 4 if self.config.if_transskip else 0,
                                                  use_batchnorm=False)
        self.up1 = SwinTransformer3D.DecoderBlock(self.config.embed_dim * 4, self.config.embed_dim * 2,
                                                  skip_channels=self.config.embed_dim * 2 if self.config.if_transskip else 0,
                                                  use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = SwinTransformer3D.DecoderBlock(self.config.embed_dim * 2, self.config.embed_dim,
                                                  skip_channels=self.config.embed_dim if self.config.if_transskip else 0,
                                                  use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = SwinTransformer3D.DecoderBlock(self.config.embed_dim, self.config.embed_dim // 2,
                                                  skip_channels=self.config.embed_dim // 2 if self.config.if_convskip else 0,
                                                  use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = SwinTransformer3D.DecoderBlock(self.config.embed_dim // 2, self.config.reg_head_chan,
                                                  skip_channels=self.config.reg_head_chan if self.config.if_convskip else 0,
                                                  use_batchnorm=False)  # 384, 160, 160, 256
        self.fc = nn.Linear(self.config.reg_head_chan * self.config.img_size[0] * self.config.img_size[1] * self.config.img_size[2], 4)

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = SwinTransformer3D.Conv3dReLU(1, self.config.embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = SwinTransformer3D.Conv3dReLU(1, self.config.reg_head_chan, 3, 1, use_batchnorm=False)

    def forward(self, x):
        if self.config.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.model(x)

        if self.config.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)

        x = x.view(x.size(0), -1)  # 展平除了 batch_size 维度以外的所有维度
        x = self.fc(x)  # 通过全连接层
        return x


data = torch.rand(1, 1, 128, 128, 128).cuda()

model = SwinTransformerClassify()
model.cuda()

while True:
    result = model.forward(data)
    print(result.cpu())
