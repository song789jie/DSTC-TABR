import torch
import numpy as np
from itertools import cycle
import intel_npu_acceleration_library
from model.ASTC import ASTC_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Codec_processing:
    def __init__(self,
                 model_path,
                 ):
        super(Codec_processing, self).__init__()

        self.load_model = torch.load(model_path, map_location=device)
        print(self.load_model.keys())
        self.model_total = ASTC_model()
        self.model_total.load_state_dict(self.load_model['model'])
        # for NPU #
        self.model_total = intel_npu_acceleration_library.compile(self.model_total, dtype=torch.float16)
        self.model_total.to(device)
        self.model_total.eval()

    def encode_return_indices_only(self, data_orig):

        x = data_orig.to(device)
        mean_raw_x = torch.mean(x, dim=-1, keepdim=True).detach()
        x = x - mean_raw_x
        std_raw_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_raw_x

        x_stationary = x.clone()

        x = x.view(-1, 1, 8, 8)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, 1, 8)

        x_embedding = self.model_total.embedding(x)
        x_8 = self.model_total.encoder_causal_conv(x_embedding)

        x_16 = x_8.view(-1, 2, 3, 4)
        x_16 = x_16.permute(0, 2, 1, 3).contiguous().view(-1, 3, 8)
        x_16_compress = self.model_total.Down_compress_4(x_16)

        x_32 = x_16_compress.view(-1, 2, 4, 4)
        x_32 = x_32.permute(0, 2, 1, 3).contiguous().view(-1, 4, 8)
        x_32_compress = self.model_total.Down_compress_5(x_32)

        x_64 = x_32_compress.view(-1, 2, 5, 4)
        x_64 = x_64.permute(0, 2, 1, 3).contiguous().view(-1, 5, 8)
        x_64_compress = self.model_total.Down_compress_6(x_64)

        xhat_64, indices_64 = self.model_total.fsq(x_64_compress)
        xhat_32, indices_32 = self.model_total.fsq(x_32_compress)
        xhat_16, indices_16 = self.model_total.fsq(x_16_compress)
        xhat_8, indices_8 = self.model_total.fsq(x_8)
        return indices_64, indices_32, indices_16, indices_8

