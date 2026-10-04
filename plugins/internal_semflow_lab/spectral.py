"""Isolated SpecTF forecast override: all text bands and internal conditioning.
Derived from the local native implementation; original file is preserved.
"""
from models.SpecTF import *

class DistributionSpecTF(FreqModelHistPred):

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, text_embeddings=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-05)
        x_enc /= stdev
        x = torch.fft.rfft(x_enc, dim=1, norm='ortho')
        x_real = self.realEmb(x.real) - self.imagEmb(x.imag)
        x_imag = self.imagEmb(x.real) + self.realEmb(x.imag)
        sample = x_real[:, 0, :, :]
        position_emb = self.position_emb(sample).unsqueeze(1)
        x_real += position_emb
        x_imag += position_emb
        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        B, N_c, T, D = x_real.shape
        x = self.internal_fuse(x, 0)
        bias = x
        text = text_embeddings
        if self.configs.fuse_history:
            attn_mask = ConstantMask(x.real.shape[0], x.real.shape[1], text_embeddings.shape[1], device=x_enc.device)
            B, N_c, H_f, D = x.shape
            B, N_c, H, H_f, D = text.shape
            x = x.reshape(B * N_c, H_f, D).contiguous()
            if self.use_all_bands:
                text_power = text.abs().mean(dim=(2, 4))
                band_weights = torch.softmax(text_power, dim=-1)
                text = (text * band_weights[:, :, None, :, None]).sum(dim=3).reshape(B * N_c, H, D)
            else:
                text = text[:, :, :, 0, :].reshape(B * N_c, H, D)
            output, attn_score = self.freq_attn_layer(x, text, text, attn_mask)
            output = output.reshape(B, N_c, H_f, D).contiguous()
            x = x.reshape(B, N_c, H_f, D).contiguous()
            if self.dominance_freq != self.H_f:
                one_mask = torch.ones(B, N_c, self.dominance_freq, self.embed_size).to(output.device)
                zero_mask = torch.ones(B, N_c, self.H_f - self.dominance_freq, self.embed_size).to(output.device)
                mask = torch.cat([one_mask, zero_mask], dim=2)
                output = output * mask
            if self.configs.use_product:
                real = torch.mul(output.real, x.real) - torch.mul(output.imag, x.imag)
                imag = torch.mul(output.real, x.imag) + torch.mul(output.imag, x.real)
            else:
                real = output.real
                imag = output.imag
            real += x.real
            imag += x.imag
            real = self.real_norm_2(real)
            imag = self.imag_norm_2(imag)
        elif self.configs.sum_fusion:
            real_text = torch.sum(text_embeddings.real, dim=2)
            imag_text = torch.sum(text_embeddings.imag, dim=2)
            real = x.real + real_text
            imag = x.imag + imag_text
        else:
            real = x.real
            imag = x.imag
        if self.configs.only_text_input:
            real = text_embeddings.real[:, :, :, 0, :]
            imag = text_embeddings.imag[:, :, :, 0, :]
        low_specxy_real = self.freq_upsampler_real(real.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) - self.freq_upsampler_imag(imag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        low_specxy_imag = self.freq_upsampler_real(imag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + self.freq_upsampler_imag(real.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = torch.stack([low_specxy_real, low_specxy_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = self.internal_fuse(x, 1)
        real, imag = (x.real, x.imag)
        x_real = self.realEmb_dec(real) - self.imagEmb_dec(imag)
        x_imag = self.imagEmb_dec(real) + self.realEmb_dec(imag)
        text = torch.stack([x_real, x_imag], dim=-1)
        text = F.softshrink(text, lambd=self.sparsity_threshold)
        text = torch.view_as_complex(text)
        x = torch.fft.irfft(text, n=self.pred_len, dim=2, norm='ortho')
        x = x * stdev.unsqueeze(-1) + means.unsqueeze(-1)
        x = x.squeeze(1)
        return x
