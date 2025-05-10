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
from model.custom_Module import Conv1DCompression, Conv1DDecompression


class ASTC_model(nn.Module):
    def __init__(
            self,
            *,
            channels=8,
            strides=(1, 2),
            channel_multi=(1, 2),
            FSQ_levels=None,
            codebook_dim=3,
            input_channels=1,
            embedding_dim=16,
            discr_multi_scales=(1, 0.5, 0.25, 0.125),
            recon_loss_weight=10.,
            adversarial_loss_weight=1.,
            feature_loss_weight=1.,
            SSIM_loss_weight=10
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

        self.fsq = FSQ(FSQ_levels)

        self.Down_compress_4 = Conv1DCompression(in_channels=3, out_channels=4)
        self.Up_compress_4 = Conv1DDecompression(in_channels=4, out_channels=3)
        self.Down_compress_5 = Conv1DCompression(in_channels=4, out_channels=5)
        self.Up_compress_5 = Conv1DDecompression(in_channels=5, out_channels=4)
        self.Down_compress_6 = Conv1DCompression(in_channels=5, out_channels=6)
        self.Up_compress_6 = Conv1DDecompression(in_channels=6, out_channels=5)

    def non_discr_parameters(self):
        return [
            # 用*返回具体的值，而不是地址
            *self.embedding.parameters(),
            *self.encoder_causal_conv.parameters(),
            *self.decoder_causal_conv.parameters(),
            *self.Down_compress_4.parameters(),
            *self.Up_compress_4.parameters(),
            *self.Down_compress_5.parameters(),
            *self.Up_compress_5.parameters(),
            *self.Down_compress_6.parameters(),
            *self.Up_compress_6.parameters(),
        ]

    def forward(
            self,
            x,
            return_recons_only=False,
            return_discr_loss=False,
            return_loss_breakdown=False,
            apply_grad_penalty=False,
            return_discr_losses_separately=False,
    ):
        raw_x = x.clone().detach()
        mean_raw_x = torch.detach(torch.mean(x, dim=-1, keepdim=True))
        x = x - mean_raw_x
        std_raw_x = torch.detach(torch.sqrt
                                 (torch.var(
                                     x, dim=-1, keepdim=True, unbiased=False) + 1e-5))
        x = x / std_raw_x

        orig_stationary_x = x.clone()
        x_input_splits_len_8 = torch.split(x, 8, dim=2)
        ###
        # print(orig_stationary_x[5, :, 8:16])
        # print(x_splits[1][5, :, :])
        ###
        x_decode_all_len_8 = []
        x_encode_all_len_8 = []
        for x_split in x_input_splits_len_8:
            x_embed_len_8 = self.embedding(x_split)
            x_encode_len_8 = self.encoder_causal_conv(x_embed_len_8)
            x_encode_all_len_8.append(x_encode_len_8)
            x_fsq_len_8, x_fsq_indices_len_8 = self.fsq(x_encode_len_8)

            x_decode_len_8 = self.decoder_causal_conv(x_fsq_len_8)
            x_decode_all_len_8.append(x_decode_len_8)
        x_hat_len_8 = torch.cat(x_decode_all_len_8, dim=2)
        recon_stationary_x_len_8 = x_hat_len_8.clone()
        x_recon_len_8 = x_hat_len_8 * std_raw_x + mean_raw_x

        x_decode_all_len_8 = []
        x_encode_all_len_16 = []
        x_encode_all_len_8A8 = torch.cat(x_encode_all_len_8, dim=2).split(8, dim=2)
        for x_encode_len_16 in x_encode_all_len_8A8:
            x_encode_len_16 = self.Down_compress_4(x_encode_len_16)
            x_encode_all_len_16.append(x_encode_len_16)
            x_fsq_len_16, x_fsq_indices_len_16 = self.fsq(x_encode_len_16)

            x_decode_len_16 = self.Up_compress_4(x_fsq_len_16)
            x_decode_len_16 = torch.split(x_decode_len_16, 4, dim=2)
            for x_decode_len_8 in x_decode_len_16:
                x_decode_len_8 = self.decoder_causal_conv(x_decode_len_8)
                x_decode_all_len_8.append(x_decode_len_8)
        x_hat_len_16 = torch.cat(x_decode_all_len_8, dim=2)
        recon_stationary_x_len_16 = x_hat_len_16.clone()
        x_recon_len_16 = x_hat_len_16 * std_raw_x + mean_raw_x

        x_decode_all_len_8 = []
        x_encode_all_len_32 = []
        x_encode_all_len_16A16 = torch.cat(x_encode_all_len_16, dim=2).split(8, dim=2)
        for x_encode_len_32 in x_encode_all_len_16A16:
            x_encode_len_32 = self.Down_compress_5(x_encode_len_32)
            x_encode_all_len_32.append(x_encode_len_32)
            x_fsq_len_32, x_fsq_indices_len_32 = self.fsq(x_encode_len_32)
            x_decode_len_32 = self.Up_compress_5(x_fsq_len_32)
            x_decode_len_16 = self.Up_compress_4(x_decode_len_32)
            x_decode_len_16 = torch.split(x_decode_len_16, 4, dim=2)
            for x_decode_len_8 in x_decode_len_16:
                x_decode_len_8 = self.decoder_causal_conv(x_decode_len_8)
                x_decode_all_len_8.append(x_decode_len_8)
        x_hat_len_32 = torch.cat(x_decode_all_len_8, dim=2)
        recon_stationary_x_len_32 = x_hat_len_32.clone()
        x_recon_len_32 = x_hat_len_32 * std_raw_x + mean_raw_x

        x_decode_all_len_8 = []
        x_encode_all_len_32A32 = torch.cat(x_encode_all_len_32, dim=2)
        x_encode_len_64 = self.Down_compress_6(x_encode_all_len_32A32)
        x_fsq_len_64, x_fsq_indices_len_64 = self.fsq(x_encode_len_64 )
        x_decode_len_64 = self.Up_compress_6(x_fsq_len_64)
        x_decode_len_32 = self.Up_compress_5(x_decode_len_64)
        x_decode_len_16 = self.Up_compress_4(x_decode_len_32)
        x_decode_len_16 = torch.split(x_decode_len_16, 4, dim=2)
        for x_decode_len_8 in x_decode_len_16:
            x_decode_len_8 = self.decoder_causal_conv(x_decode_len_8)
            x_decode_all_len_8.append(x_decode_len_8)

        x_hat_len_64 = torch.cat(x_decode_all_len_8, dim=2)
        recon_stationary_x_len_64 = x_hat_len_64.clone()
        x_recon_len_64 = x_hat_len_64 * std_raw_x + mean_raw_x

        if return_recons_only:
            return x_recon_len_8, x_recon_len_16, x_recon_len_32, x_recon_len_64

        if return_discr_loss:
            real = orig_stationary_x
            fakes = [
                recon_stationary_x_len_8.detach().requires_grad_(True),
                recon_stationary_x_len_16.detach().requires_grad_(True),
                recon_stationary_x_len_32.detach().requires_grad_(True),
                recon_stationary_x_len_64.detach().requires_grad_(True)
            ]
            discr_losses = []
            discr_grad_penalties = []
            scaled_real = real.requires_grad_(True)
            scaled_fakes = fakes

            for i, downsample in enumerate(self.downsamples):
                for j in range(i + 1, 4):
                    scaled_fakes[j] = downsample(scaled_fakes[j])

            for discr, downsample, scaled_fake in zip(self.discriminators, self.downsamples, scaled_fakes):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))
                (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(
                    partial(discr, return_intermediates=True), (scaled_real, scaled_fake))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
                    discr_grad_penalties.append(gradient_penalty(scaled_real, one_discr_loss))

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()
                return all_discr_losses

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in
                                     zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend(
                [(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in
                 zip(self.discr_multi_scales, discr_grad_penalties)])

            return discr_losses_pkg

        recon_loss_64 = F.mse_loss(orig_stationary_x, recon_stationary_x_len_64)
        recon_loss_32 = F.mse_loss(orig_stationary_x, recon_stationary_x_len_32)
        recon_loss_16 = F.mse_loss(orig_stationary_x, recon_stationary_x_len_16)
        recon_loss_8 = F.mse_loss(orig_stationary_x, recon_stationary_x_len_8)

        SSIM_loss_64 = SSIM_1D_loss(recon_stationary_x_len_64, orig_stationary_x)
        SSIM_loss_32 = SSIM_1D_loss(recon_stationary_x_len_32, orig_stationary_x)
        SSIM_loss_16 = SSIM_1D_loss(recon_stationary_x_len_16, orig_stationary_x)
        SSIM_loss_8 = SSIM_1D_loss(recon_stationary_x_len_8, orig_stationary_x)

        # recon_loss = recon_loss_64
        # SSIM_loss = SSIM_loss_64
        recon_loss = (recon_loss_8 + recon_loss_16 + recon_loss_32 + recon_loss_64) / 4
        SSIM_loss = (SSIM_loss_8 + SSIM_loss_16 + SSIM_loss_32 + SSIM_loss_64) / 4
        adversarial_losses = []

        discr_intermediates = []

        scaled_real = orig_stationary_x
        scaled_fakes = [recon_stationary_x_len_8,
                        recon_stationary_x_len_16,
                        recon_stationary_x_len_32,
                        recon_stationary_x_len_64]
        # scaled_fakes = [recon_stationary_x_len_8,
        #                 recon_stationary_x_len_8,
        #                 recon_stationary_x_len_8,
        #                 recon_stationary_x_len_8]

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

        total_loss = ((recon_loss * self.recon_loss_weight) +
                      (adversarial_loss * self.adversarial_loss_weight) +
                      (feature_loss * self.feature_loss_weight) +
                      (SSIM_loss * self.SSIM_loss_weight))

        if return_loss_breakdown:
            return total_loss, (recon_loss, adversarial_loss, feature_loss, SSIM_loss,
                                recon_loss_64, recon_loss_32, recon_loss_16, recon_loss_8)

        return total_loss


if __name__ == '__main__':
    model = ASTC_model()
    test = model(torch.randn(16, 1, 64))
