# import torch
# import torch.nn as nn
#
# from lib.utils.token_utils import patch2token
#
#
# class CFN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.out_channels = out_channels
#         self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#
#         #
#         self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
#                                 groups=in_channels)
#         self.bn33 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
#
#         self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
#                                 groups=in_channels)
#         self.bn11 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
#
#         #
#         self.conv_up = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1,
#                                  padding=0)
#         self.bn_up = nn.BatchNorm2d(in_channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
#         self.act = nn.GELU()
#
#         self.conv_down = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1,
#                                    padding=0)
#         self.bn_down = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
#
#         # down
#         self.adjust = nn.Conv2d(in_channels, out_channels, 1)
#
#         # norm all
#         self.norm = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x, H, W):
#         B, N, _C = x.shape
#
#         x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
#         # print(f"x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous():{x.shape}") ([2, 768, 16, 16])
#         residual = self.residual(x)
#
#         #  + skip-connection
#         x = x + self.bn11(self.conv11(x)) + self.bn33(self.conv33(x))
#
#         #  + skip-connection
#         x = x + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(x)))))
#
#         x = self.adjust(x)
#
#         out = self.norm(residual + x)
#         return out
#
# class MS_Fusion(nn.Module):
#     def __init__(self, dim=8, xavier_init=False):
#         super().__init__()
#
#         self.adapter_down1 = nn.Linear(768, dim)
#         self.adapter_down2 = nn.Linear(768, dim)
#         self.adapter_mid_x = CFN(dim,dim)
#         self.adapter_mid_z = CFN(dim,dim)
#         self.adapter_up1 = nn.Linear(dim, 768)
#         self.adapter_up2 = nn.Linear(dim, 768)
#
#         #nn.init.xavier_uniform_(self.adapter_down.weight)
#         # nn.init.zeros_(self.adapter_mid.bias)
#         # nn.init.zeros_(self.adapter_mid.weight)
#
#         # nn.init.zeros_(self.adapter_mid_upscale.bias)
#         # nn.init.zeros_(self.adapter_mid_upscale.weight)
#         nn.init.zeros_(self.adapter_up1.weight)
#         nn.init.zeros_(self.adapter_up2.weight)
#         nn.init.zeros_(self.adapter_up1.bias)
#         nn.init.zeros_(self.adapter_up2.bias)
#         nn.init.zeros_(self.adapter_down1.weight)
#         nn.init.zeros_(self.adapter_down2.weight)
#         nn.init.zeros_(self.adapter_down2.bias)
#         nn.init.zeros_(self.adapter_down1.bias)
#
#         #self.act = QuickGELU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#
#     def forward(self, x):
#         # B, N, C = x.shape
#         z_x = x[:,:64,:]
#         x_x = x[:,64:,:]
#
#         x_down = self.adapter_down1(x_x)
#         x_down = self.adapter_mid_x(x_down,16,16)
#         x_down = patch2token(x_down)
#         x_down = self.dropout(x_down)
#         x_up = self.adapter_up1(x_down)
#
#         z_down = self.adapter_down2(z_x)
#         z_down = self.adapter_mid_z(z_down, 8, 8)
#         z_down = patch2token(z_down)
#         z_down = self.dropout(z_down)
#         z_up = self.adapter_up2(z_down)
#
#         x = torch.cat((z_up,x_up),dim=1)
#         return x
#
#
# # x = torch.ones([2,320,768])
# # model = MS_Fusion()
# # xo = model(x)
# # print(xo.shape)   # torch.Size([2, 768, 16, 16])


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
import time


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        query_layer = QueryLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers, query_layer)
        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, src_search, pos_temp, pos_search, query_embed):
        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        search_shape = src_search.shape
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, search_shape[0], 1)

        # memory_temp, memory_search, torch_s, torch_e = self.encoder(src1=src_temp, src2=src_search,
        #                                           pos_src1=pos_temp,
        #                                           pos_src2=pos_search)
        memory_temp, memory_search= self.encoder(src1=src_temp, src2=src_search,
                                                 pos_src1=pos_temp,
                                                 pos_src2=pos_search,
                                                 query_embed=query_embed)
        hs = self.decoder(memory_search, memory_temp,
                          pos_enc=query_embed, pos_dec=pos_search)
        # hs = self.decoder(src_search, src_temp,
        #                   pos_enc=pos_temp, pos_dec=pos_search)
        # return hs.unsqueeze(0).transpose(1, 2), torch_s, torch_e
        return hs.unsqueeze(0).transpose(1, 2)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers, query_layer):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.query_layer = query_layer
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None,
                query_embed = None):
        output1 = src1
        output2 = src2

        output1 = self.query_layer(output1, query_embed, src1_mask=src1_mask,
                                   src1_key_padding_mask=src1_key_padding_mask,
                                   pos_src1=pos_src1)

        for layer in self.layers:
            # output1, output2, torch_s, torch_e = layer(output1, output2, src1_mask=src1_mask,
            #                          src2_mask=src2_mask,
            #                          src1_key_padding_mask=src1_key_padding_mask,
            #                          src2_key_padding_mask=src2_key_padding_mask,
            #                          pos_src1=pos_src1, pos_src2=pos_src2)
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=query_embed, pos_src2=pos_src2)

        # return output1, output2, torch_s, torch_e
        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        # self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        # self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        # self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        # self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        # q1 = k1 = self.with_pos_embed(src1, pos_src1)
        # src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
        #                        key_padding_mask=src1_key_padding_mask)[0]
        # src1 = src1 + self.dropout11(src12)
        # src1 = self.norm11(src1)
        #
        # q2 = k2 = self.with_pos_embed(src2, pos_src2)
        # src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
        #                        key_padding_mask=src2_key_padding_mask)[0]
        # src2 = src2 + self.dropout21(src22)
        # src2 = self.norm21(src2)
        # torch_s = time.time()

        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        # torch_e = time.time()

        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2, attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)




        # return src1, src2, torch_s, torch_e
        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)

class QueryLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):

        src2 = self.multihead_attn(query=self.with_pos_embed(src2, pos_src2),
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]
        src2 = src2 + self.dropout(src2)
        src2 = self.norm(src2)
        return src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")