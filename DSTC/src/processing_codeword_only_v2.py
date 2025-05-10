import torch
import numpy as np
from itertools import cycle
# import intel_npu_acceleration_library
from model.ASTC import ASTC_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Codec_processing_init:
    def __init__(self,
                 model_path,
                 Bit_width=0
                 ):
        super(Codec_processing_init, self).__init__()

        self.load_model = torch.load(model_path, map_location=device)
        print(self.load_model.keys())
        self.model_total = ASTC_model()
        self.model_total.load_state_dict(self.load_model['model'])
        # for NPU #
        # self.model_total = intel_npu_acceleration_library.compile(self.model_total, dtype=torch.float16)
        self.model_total.to(device)
        self.model_total.eval()
        self.Bit_width = Bit_width
        if self.Bit_width != 0:
            self.p_set = self.build_p()

    def encode_return_indices_only(self, data_orig):

        x = data_orig.to(device)
        mean_raw_x = torch.mean(x, dim=-1, keepdim=True).detach()
        x = x - mean_raw_x
        std_raw_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_raw_x

        x_input_splits_len_8 = torch.split(x, 8, dim=2)
        ###
        # print(orig_stationary_x[5, :, 8:16])
        # print(x_splits[1][5, :, :])
        ###

        x_encode_all_len_8 = []
        x_indices_all_len_8 = []
        for x_split in x_input_splits_len_8:
            x_embed_len_8 = self.model_total.embedding(x_split)
            x_encode_len_8 = self.model_total.encoder_causal_conv(x_embed_len_8)
            x_encode_all_len_8.append(x_encode_len_8)
            x_fsq_len_8, x_fsq_indices_len_8 = self.model_total.fsq(x_encode_len_8)
            # print(x_fsq_indices_len_8.shape)
            x_indices_all_len_8.append(x_fsq_indices_len_8)
        indices_8 = torch.cat(x_indices_all_len_8, dim=1)

        x_encode_all_len_16 = []
        x_indices_all_len_16 = []
        x_encode_all_len_8A8 = torch.cat(x_encode_all_len_8, dim=2).split(8, dim=2)
        for x_encode_len_16 in x_encode_all_len_8A8:
            x_encode_len_16 = self.model_total.Down_compress_4(x_encode_len_16)
            x_encode_all_len_16.append(x_encode_len_16)
            x_fsq_len_16, x_fsq_indices_len_16 = self.model_total.fsq(x_encode_len_16)
            x_indices_all_len_16.append(x_fsq_indices_len_16)
        indices_16 = torch.cat(x_indices_all_len_16, dim=1)

        x_encode_all_len_32 = []
        x_indices_all_len_32 = []
        x_encode_all_len_16A16 = torch.cat(x_encode_all_len_16, dim=2).split(8, dim=2)
        for x_encode_len_32 in x_encode_all_len_16A16:
            x_encode_len_32 = self.model_total.Down_compress_5(x_encode_len_32)
            x_encode_all_len_32.append(x_encode_len_32)
            x_fsq_len_32, x_fsq_indices_len_32 = self.model_total.fsq(x_encode_len_32)
            x_indices_all_len_32.append(x_fsq_indices_len_32)
        indices_32 = torch.cat(x_indices_all_len_32, dim=1)

        x_encode_all_len_32A32 = torch.cat(x_encode_all_len_32, dim=2)
        x_encode_len_64 = self.model_total.Down_compress_6(x_encode_all_len_32A32)
        x_fsq_len_64, x_fsq_indices_len_64 = self.model_total.fsq(x_encode_len_64)
        indices_64 = x_fsq_indices_len_64

        return indices_64, indices_32, indices_16, indices_8

    def encode_return_residuals_only(self, data_orig):

        x = data_orig.to(device)
        mean_raw_x = torch.mean(x, dim=-1, keepdim=True).detach()
        x = x - mean_raw_x
        std_raw_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_raw_x
        x_stationary = x.clone()
        x_input_splits_len_8 = torch.split(x, 8, dim=2)
        ###
        # print(orig_stationary_x[5, :, 8:16])
        # print(x_splits[1][5, :, :])
        ###

        x_encode_all_len_8 = []
        x_indices_all_len_8 = []
        for x_split in x_input_splits_len_8:
            x_embed_len_8 = self.model_total.embedding(x_split)
            x_encode_len_8 = self.model_total.encoder_causal_conv(x_embed_len_8)
            x_encode_all_len_8.append(x_encode_len_8)
            x_fsq_len_8, x_fsq_indices_len_8 = self.model_total.fsq(x_encode_len_8)
            # print(x_fsq_indices_len_8.shape)
            x_indices_all_len_8.append(x_fsq_indices_len_8)
        indices_8 = torch.cat(x_indices_all_len_8, dim=1)

        x_encode_all_len_16 = []
        x_indices_all_len_16 = []
        x_encode_all_len_8A8 = torch.cat(x_encode_all_len_8, dim=2).split(8, dim=2)
        for x_encode_len_16 in x_encode_all_len_8A8:
            x_encode_len_16 = self.model_total.Down_compress_4(x_encode_len_16)
            x_encode_all_len_16.append(x_encode_len_16)
            x_fsq_len_16, x_fsq_indices_len_16 = self.model_total.fsq(x_encode_len_16)
            x_indices_all_len_16.append(x_fsq_indices_len_16)
        indices_16 = torch.cat(x_indices_all_len_16, dim=1)

        x_encode_all_len_32 = []
        x_indices_all_len_32 = []
        x_encode_all_len_16A16 = torch.cat(x_encode_all_len_16, dim=2).split(8, dim=2)
        for x_encode_len_32 in x_encode_all_len_16A16:
            x_encode_len_32 = self.model_total.Down_compress_5(x_encode_len_32)
            x_encode_all_len_32.append(x_encode_len_32)
            x_fsq_len_32, x_fsq_indices_len_32 = self.model_total.fsq(x_encode_len_32)
            x_indices_all_len_32.append(x_fsq_indices_len_32)
        indices_32 = torch.cat(x_indices_all_len_32, dim=1)

        x_encode_all_len_32A32 = torch.cat(x_encode_all_len_32, dim=2)
        x_encode_len_64 = self.model_total.Down_compress_6(x_encode_all_len_32A32)
        x_fsq_len_64, x_fsq_indices_len_64 = self.model_total.fsq(x_encode_len_64)
        indices_64 = x_fsq_indices_len_64

        x_fsq_len_64_all = self.model_total.fsq.indices_to_codes(indices_64)
        x_fsq_len_32_all = self.model_total.fsq.indices_to_codes(indices_32)
        x_fsq_len_16_all = self.model_total.fsq.indices_to_codes(indices_16)
        x_fsq_len_8_all = self.model_total.fsq.indices_to_codes(indices_8)
        # print(x_fsq_len_64.shape, x_fsq_len_32.shape, x_fsq_len_16.shape, x_fsq_len_8.shape)

        x_hat_all_len_64 = []
        x_decode_len_64 = self.model_total.Up_compress_4(
            self.model_total.Up_compress_5(
                self.model_total.Up_compress_6(x_fsq_len_64_all)))
        x_decode_len_8_all = torch.split(x_decode_len_64, 4, dim=2)
        for x_decode_len_8 in x_decode_len_8_all:
            x_decode_len_8 = self.model_total.decoder_causal_conv(x_decode_len_8)
            x_hat_all_len_64.append(x_decode_len_8)
        x_hat_len_64 = torch.cat(x_hat_all_len_64, dim=2)

        x_hat_all_len_32 = []
        # print(x_fsq_len_32_all.shape)
        x_fsq_len_32_all = torch.split(x_fsq_len_32_all, 5, dim=1)
        for x_fsq_len_32 in x_fsq_len_32_all:
            x_decode_len_32 = self.model_total.Up_compress_4(
                self.model_total.Up_compress_5(x_fsq_len_32))
            x_decode_len_8_all = torch.split(x_decode_len_32, 4, dim=2)
            for x_decode_len_8 in x_decode_len_8_all:
                x_decode_len_8 = self.model_total.decoder_causal_conv(x_decode_len_8)
                x_hat_all_len_32.append(x_decode_len_8)

        x_hat_len_32 = torch.cat(x_hat_all_len_32, dim=2)
        x_hat_all_len_16 = []
        x_fsq_len_16_all = torch.split(x_fsq_len_16_all, 4, dim=1)
        for x_fsq_len_16 in x_fsq_len_16_all:
            x_decode_len_16 = self.model_total.Up_compress_4(x_fsq_len_16)
            x_decode_len_8_all = torch.split(x_decode_len_16, 4, dim=2)
            for x_decode_len_8 in x_decode_len_8_all:
                x_decode_len_8 = self.model_total.decoder_causal_conv(x_decode_len_8)
                x_hat_all_len_16.append(x_decode_len_8)

        x_hat_len_16 = torch.cat(x_hat_all_len_16, dim=2)

        x_hat_all_len_8 = []
        x_decode_len_8_all = torch.split(x_fsq_len_8_all, 3, dim=1)

        for x_decode_len_8 in x_decode_len_8_all:
            x_decode_len_8 = self.model_total.decoder_causal_conv(x_decode_len_8)
            x_hat_all_len_8.append(x_decode_len_8)

        x_hat_len_8 = torch.cat(x_hat_all_len_8, dim=2)

        return x_stationary - x_hat_len_64, x_stationary - x_hat_len_32, \
               x_stationary - x_hat_len_16, x_stationary - x_hat_len_8

    def build_p(self):
        p_0 = []
        gamma = 4

        for i in range(2 ** self.Bit_width):
            p_0.append(2 ** (i / gamma) - 1)

        R = []
        p_0.sort()
        for a in p_0:
            R.append(a)

        R = torch.Tensor(list(R))
        R = R.mul(1.0 / torch.max(R))
        # print(R)
        return R

    def residual_non_uniform_quant(self, tensor):
        def p_quant(x, value_s):
            shape = x.shape
            x_hat = x.view(-1)
            sign = x.sign()
            value_s = value_s.type_as(x)
            x_hat = x_hat.abs()
            idexes = (x_hat.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            x_hat = value_s[idexes].view(shape).mul(sign)
            x_hat = x_hat
            x_hat = (x_hat - x).detach() + x
            return x_hat, idexes

        data = tensor
        data = data.clamp(-1, 1)
        data_q, idxs = p_quant(data, self.p_set)
        data_q = data_q

        return data_q, idxs

    def residual_non_uniform_dequant(self, idxs):
        def p_dequant(idxs, value_s):
            idxs = torch.tensor(idxs)
            shape = idxs.shape
            x_hat = value_s[idxs].view(shape)
            return x_hat

        data_q = p_dequant(idxs, self.p_set)
        data_q = data_q * 2.0

        return data_q
