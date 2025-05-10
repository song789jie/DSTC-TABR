import torch
from torch import nn
import pickle
from vector_quantize_pytorch import FSQ
from functools import partial, wraps
import torch.nn.functional as F

from model.custom_Module import (MultiScaleDiscriminator, DataEmbedding,
                                 CausalConv1d, EncoderBlock, DecoderBlock)
from model.custom_Loss import hinge_discr_loss, hinge_gen_loss, SSIM_1D_loss
from src.utils import exists, gradient_penalty


class ASTC_model(nn.Module):
    def __init__(
            self,
            *,
            channels=16,
            strides=(1, 2),
            channel_multi=(1, 2),
            FSQ_levels=None,
            codebook_dim=3,
            input_channels=1,
            embedding_dim=16,
            discr_multi_scales=(1, 0.5, 0.25, 0.125),
            recon_loss_weight=10.,
            adversarial_loss_weight=1.,
            feature_loss_weight=10.,
            SSIM_loss_weight=0.1
    ):
        super().__init__()

        if FSQ_levels is None:
            FSQ_levels = [8, 5, 5, 5]
        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        self.recon_loss_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.SSIM_loss_weight = SSIM_loss_weight

        self.single_channel = input_channels == 1
        self.strides = strides
        self.embedding_dim = embedding_dim

        layer_channels = tuple(map(lambda t: t * channels, channel_multi))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = nn.ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])

        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = nn.ModuleList(
            [nn.Identity()] + [nn.AvgPool1d(2 * factor, stride=factor, padding=factor) for factor in discr_rel_factors])
        self.x_16_conv = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)

        self.x_32_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=4, out_channels=5, kernel_size=3, stride=2, padding=1)
        )

        self.x_64_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=4, out_channels=5, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=5, out_channels=6, kernel_size=3, stride=2, padding=1)
        )
        self.embedding = DataEmbedding(c_in=1, d_model=self.embedding_dim)

        encoder_causal_conv_blocks = []
        decoder_causal_conv_blocks = []
        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_causal_conv_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride,
                                                           False, 'constant'))

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_causal_conv_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride,
                                                           False, 'constant'))

        self.encoder_causal_conv = nn.Sequential(
            CausalConv1d(self.embedding_dim, channels, 7, pad_mode='constant'),
            *encoder_causal_conv_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3, pad_mode='constant')
        )

        self.decoder_causal_conv = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 3, pad_mode='constant'),
            *decoder_causal_conv_blocks,
            CausalConv1d(channels, 1, 7, pad_mode='constant')
        )

        self.decoder_causal_conv_16 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=4, out_channels=3,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            CausalConv1d(codebook_dim, layer_channels[-1], 3, pad_mode='constant'),
            *decoder_causal_conv_blocks,
            CausalConv1d(channels, 1, 7, pad_mode='constant')
        )

        self.decoder_causal_conv_32 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=5, out_channels=4,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose1d(in_channels=4, out_channels=3,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),

            CausalConv1d(codebook_dim, layer_channels[-1], 3, pad_mode='constant'),
            *decoder_causal_conv_blocks,
            CausalConv1d(channels, 1, 7, pad_mode='constant')
        )

        self.decoder_causal_conv_64 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=6, out_channels=5,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose1d(in_channels=5, out_channels=4,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose1d(in_channels=4, out_channels=3,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),

            CausalConv1d(codebook_dim, layer_channels[-1], 3, pad_mode='constant'),
            *decoder_causal_conv_blocks,
            CausalConv1d(channels, 1, 7, pad_mode='constant')
        )

        self.fsq = FSQ(FSQ_levels)


    def non_discr_parameters(self):
        return [
            # 用*返回具体的值，而不是地址
            *self.embedding.parameters(),
            *self.x_16_conv.parameters(),
            *self.x_32_conv.parameters(),
            *self.x_64_conv.parameters(),
            *self.decoder_causal_conv_16.parameters(),
            *self.decoder_causal_conv_32.parameters(),
            *self.decoder_causal_conv_64.parameters(),
            *self.encoder_causal_conv.parameters(),
            *self.decoder_causal_conv.parameters(),
        ]

    def forward(
            self,
            x,
            return_loss_breakdown=False,
    ):
        raw_x = x.clone().detach()
        mean_raw_x = torch.detach(torch.mean(x, dim=-1, keepdim=True))
        x = x - mean_raw_x
        std_raw_x = torch.detach(torch.sqrt
                                 (torch.var(
                                     x, dim=-1, keepdim=True, unbiased=False) + 1e-5))
        x = x / std_raw_x

        orig_stationary_x = x.clone()

        x = x.view(-1, 1, 8, 8)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, 1, 8)

        x_embedding = self.embedding(x)

        x_8 = self.encoder_causal_conv(x_embedding)
        ''' x_8 [batch*8, 3, 4] '''
        x_16 = x_8.view(-1, 2, 3, 4)
        x_16 = x_16.permute(0, 2, 1, 3).contiguous().view(-1, 3, 8)
        ''' x_16 [batch*4, 3, 8] '''
        x_16 = self.x_16_conv(x_16)

        x_32 = x_8.view(-1, 4, 3, 4)
        x_32 = x_32.permute(0, 2, 1, 3).contiguous().view(-1, 3, 16)
        ''' x_32 [batch*2, 3, 16] '''
        x_32 = self.x_32_conv(x_32)

        x_64 = x_8.view(-1, 8, 3, 4)
        x_64 = x_64.permute(0, 2, 1, 3).contiguous().view(-1, 3, 32)
        ''' x_64 [batch*2, 3, 32] '''
        x_64 = self.x_64_conv(x_64)

        xhat_64, indices_64 = self.fsq(x_64)
        xhat_32, indices_32 = self.fsq(x_32)
        xhat_16, indices_16 = self.fsq(x_16)
        xhat_8, indices_8 = self.fsq(x_8)

        xhat_64 = self.decoder_causal_conv_64(xhat_64)
        xhat_32 = self.decoder_causal_conv_32(xhat_32)
        xhat_16 = self.decoder_causal_conv_16(xhat_16)
        xhat_8 = self.decoder_causal_conv(xhat_8)

        xhat_64 = xhat_64.view(-1, 1, 64)
        xhat_32 = xhat_32.view(-1, 1, 64)
        xhat_16 = xhat_16.view(-1, 1, 64)
        xhat_8 = xhat_8.view(-1, 1, 64)

        recon_stationary_x_8 = xhat_8.clone()
        x_recon_8 = xhat_8 * std_raw_x + mean_raw_x

        recon_stationary_x_16 = xhat_16.clone()
        x_recon_16 = xhat_16 * std_raw_x + mean_raw_x

        recon_stationary_x_32 = xhat_32.clone()
        x_recon_32 = xhat_32 * std_raw_x + mean_raw_x

        recon_stationary_x_64 = xhat_64.clone()
        x_recon_64 = xhat_64 * std_raw_x + mean_raw_x

        recon_loss_8 = F.mse_loss(orig_stationary_x, recon_stationary_x_8)
        recon_loss_16 = F.mse_loss(orig_stationary_x, recon_stationary_x_8)
        recon_loss_32 = F.mse_loss(orig_stationary_x, recon_stationary_x_8)
        recon_loss_64 = F.mse_loss(orig_stationary_x, recon_stationary_x_8)

        SSIM_loss_8 = SSIM_1D_loss(recon_stationary_x_8, orig_stationary_x)
        SSIM_loss_16 = SSIM_1D_loss(recon_stationary_x_16, orig_stationary_x)
        SSIM_loss_32 = SSIM_1D_loss(recon_stationary_x_32, orig_stationary_x)
        SSIM_loss_64 = SSIM_1D_loss(recon_stationary_x_64, orig_stationary_x)

        recon_loss = recon_loss_8 + recon_loss_16 + recon_loss_32 + recon_loss_64
        SSIM_loss = SSIM_loss_8 + SSIM_loss_16 + SSIM_loss_32 + SSIM_loss_64
        adversarial_losses = []

        discr_intermediates = []

        scaled_real = orig_stationary_x
        scaled_fakes = [recon_stationary_x_8,
                        recon_stationary_x_16,
                        recon_stationary_x_32,
                        recon_stationary_x_64]

        for i, downsample in enumerate(self.downsamples):
            for j in range(i+1, 4):
                scaled_fakes[j] = downsample(scaled_fakes[j])

        for discr, downsample, scaled_fake in zip(self.discriminators, self.downsamples, scaled_fakes):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))
            (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(
                partial(discr, return_intermediates=True), (scaled_real, scaled_fake))

            discr_intermediates.append((real_intermediates, fake_intermediates))

            one_adversarial_loss = hinge_gen_loss(fake_logits)
            adversarial_losses.append(one_adversarial_loss)

        feature_losses = []

        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) for real_intermediate, fake_intermediate
                      in zip(real_intermediates, fake_intermediates)]
            feature_losses.extend(losses)

        feature_loss = torch.stack(feature_losses).mean()
        adversarial_loss = torch.stack(adversarial_losses).mean()

        total_loss = (recon_loss * self.recon_loss_weight) \
                     + (adversarial_loss * self.adversarial_loss_weight) \
                     + (feature_loss * self.feature_loss_weight) \
                     + (SSIM_loss * self.SSIM_loss_weight)

        if return_loss_breakdown:
            return total_loss, (recon_loss, adversarial_loss, feature_loss, SSIM_loss)

        return total_loss


if __name__ == '__main__':
    model = ASTC_model()
    test = model(torch.randn(16, 1, 64))
