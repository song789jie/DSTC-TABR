import torch
import numpy as np
import intel_npu_acceleration_library
from model.ASTC import ASTC_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Codec_processing:
    def __init__(self,
                 model_path,
                 huffman_codebook_path,
                 ):
        super(Codec_processing, self).__init__()

        self.load_model = torch.load(model_path, map_location=device)
        self.model_total = ASTC_model()
        self.model_total.load_state_dict(self.load_model['model'])
        # for NPU #
        self.model_total = intel_npu_acceleration_library.compile(self.model_total, dtype=torch.float16)
        self.model_total.to(device)
        self.model_total.eval()
        self.huffman_codec = np.load(huffman_codebook_path, allow_pickle=True).item()

    def encode_all_type(self, data_orig):

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

        return (indices_64, indices_32, indices_16, indices_8,
                mean_raw_x, std_raw_x, x_stationary)

    def decode_all_type(self, codewords_64, codewords_32, codewords_16, codewords_8,
                           mean, std):
        # FSQ: Codewords to Representation
        xhat_64 = self.model_total.fsq.indices_to_codes(codewords_64)
        xhat_32 = self.model_total.fsq.indices_to_codes(codewords_32)
        xhat_16 = self.model_total.fsq.indices_to_codes(codewords_16)
        xhat_8 = self.model_total.fsq.indices_to_codes(codewords_8)

        xhat_64 = self.model_total.Up_compress_4(
                  self.model_total.Up_compress_5(
                  self.model_total.Up_compress_6(xhat_64)))
        xhat_32 = self.model_total.Up_compress_4(
                  self.model_total.Up_compress_5(xhat_32))
        xhat_16 = self.model_total.Up_compress_4(xhat_16)

        xhat_64 = xhat_64.view(-1, 3, 2, 4)
        xhat_64 = xhat_64.permute(0, 2, 1, 3).contiguous()
        xhat_64 = xhat_64.view(-1, 3, 4)

        xhat_32 = xhat_32.view(-1, 3, 2, 4)
        xhat_32 = xhat_32.permute(0, 2, 1, 3).contiguous()
        xhat_32 = xhat_32.view(-1, 3, 4)

        xhat_16 = xhat_16.view(-1, 3, 2, 4)
        xhat_16 = xhat_16.permute(0, 2, 1, 3).contiguous()
        xhat_16 = xhat_16.view(-1, 3, 4)

        # Transpose Convolution Layers
        xhat_64 = self.model_total.decoder_causal_conv(xhat_64)
        xhat_32 = self.model_total.decoder_causal_conv(xhat_32)
        xhat_16 = self.model_total.decoder_causal_conv(xhat_16)
        xhat_8 = self.model_total.decoder_causal_conv(xhat_8)

        xhat_64 = xhat_64.view(-1, 1, 64)
        xhat_32 = xhat_32.view(-1, 1, 64)
        xhat_16 = xhat_16.view(-1, 1, 64)
        xhat_8 = xhat_8.view(-1, 1, 64)

        x_recon_64 = xhat_64 * mean + std
        x_recon_32 = xhat_32 * mean + std
        x_recon_16 = xhat_16 * mean + std
        x_recon_8 = xhat_8 * mean + std

        return x_recon_64, x_recon_32, x_recon_16, x_recon_8

    def FSQ_codewords_huffman_encode(self, indices):
        len_bits = 0
        string_bits = ''
        for j in range(len(indices)):
            for i in range(len(indices[j])):
                indices_int = indices[j][i].item()
                bits = self.huffman_codec.get(indices_int)
                if bits is None:
                    print(indices_int)
                string_bits = string_bits + bits
                len_bits = len_bits + len(bits)
        return string_bits, len_bits

    def FSQ_codewords_huffman_decode(self, string_bits, num_index):
        decoded_indices = torch.zeros(num_index).type(torch.int32).to(device)
        flipped_dict = {value: key for key, value in self.huffman_codec.items()}
        index = 0
        current_code = ""
        for bit in string_bits:
            current_code += bit
            char = flipped_dict.get(current_code)
            if char is None:
                continue
            else:
                if index >= num_index:
                    decoded_indices = torch.cat((decoded_indices, torch.Tensor([char])))
                    index = index + 1
                    current_code = ""
                else:
                    decoded_indices[index] = char
                    index = index + 1
                    current_code = ""
        decoded_indices = torch.unsqueeze(decoded_indices, dim=0)
        return decoded_indices