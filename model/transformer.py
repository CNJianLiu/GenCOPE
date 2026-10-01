# Copyright (c) UWA CSSE. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional
from numpy import block

import torch
import torch.nn.functional as F
from torch import Tensor, nn




class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x

class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model
        
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        x = tgt

        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 batch_first=False):
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src:Tensor,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        if self.batch_first:
            src = src.permute(1, 0, 2)
            pos = pos.permute(1, 0, 2) if pos is not None else None
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.batch_first:
            src = src.permute(1, 0, 2)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        if self.batch_first:
            src = src.permute(1, 0, 2)
            pos = pos.permute(1, 0, 2)
            
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.batch_first:
            src = src.permute(1, 0, 2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 batch_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.batch_first = batch_first
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.batch_first:
            tgt = tgt.permute(1, 0, 2)
            memory = memory.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2) if query_pos is not None else None
            
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.batch_first:
            tgt = tgt.permute(1, 0, 2)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        if self.batch_first:
            tgt = tgt.permute(1, 0, 2)
            memory = memory.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2) if query_pos is not None else None
            
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if self.batch_first:
            tgt = tgt.permute(1, 0, 2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
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



from torch.nn import MultiheadAttention
from copy import deepcopy


class CoTransformerHalfLayer(nn.Module):
    def __init__(self,
                 d_model:int=256,
                 nhead:int=4,
                 d_ffn:int=1024,
                 device:str='cuda',
                 dropout:float=0.1,
                 activation:str='relu',
                 batch_first:bool=True) -> None:
        super().__init__()
        self.device = device
        self.d_model = d_model 
        self.nhead = nhead
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.batch_first = batch_first
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        
        self.mha = MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True,
            device=self.device
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ffn),
            self.activation,
            nn.Linear(self.d_ffn, d_model)
        ).to(self.device)
        self.normlayer1 = nn.LayerNorm(self.d_model).to(self.device)
        self.normlayer2 = nn.LayerNorm(self.d_model).to(self.device)
        
    def forward(self, query:Tensor, kv:Tensor, mask:Tensor=None, attn_mask:Tensor=None):
        x,_ = self.mha.forward(query=query, key=kv, value=kv, key_padding_mask=mask, attn_mask=attn_mask)
        x = self.normlayer1(query + x)
        x = self.normlayer2(x + self.ffn.forward(x))
        return x
    
class CoTransformerLayer(nn.Module):
    def __init__(self,
                 d_model:int=256,
                 nhead:int=4,
                 d_ffn:int=1024,
                 device:str='cuda',
                 dropout:float=0.1,
                 activation:str='relu',
                 batch_first:bool=True) -> None:
        super().__init__()
        self.LeftCoTRMLayer = CoTransformerHalfLayer(
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            device=device,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        self.RightCoTRMLayer = deepcopy(self.LeftCoTRMLayer)
        
        self.LeftTRMLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=dropout,
            device=device,
            activation=activation,
            batch_first=batch_first
        )
        
        self.RightTRMLayer = deepcopy(self.LeftTRMLayer)
        
    def forward(self, 
                left: Tensor, 
                right:Tensor, 
                left_mask:Tensor=None,
                right_mask:Tensor=None,
                left_attn_mask:Tensor=None,
                right_attn_mask:Tensor=None
                ):
        """
        Args:
            left (Tensor): [B, Ll, E]
            right (Tensor): [B, Lr, E]
            left_mask (Tensor): [B, Ll] or None, it is bool
            right_mask (Tensor): [B, Lr] or None, it is bool
        Returns:
            left, right
        """
        left = self.LeftCoTRMLayer.forward(
            query=left,
            kv=right,
            mask=left_mask,
            attn_mask=left_attn_mask
        )
        left = self.LeftTRMLayer.forward(src=left,
                                         src_key_padding_mask=left_mask,
                                         src_mask=left_attn_mask)
        
        right = self.RightCoTRMLayer.forward(
            query=right,
            kv=left,
            mask=right_mask,
            attn_mask=right_attn_mask
        )
        
        right = self.RightTRMLayer.forward(src=right,
                                           src_key_padding_mask=right_mask,
                                           src_mask=right_attn_mask)
        
        return left, right
        

class ParallelTransformer(nn.Module):
    def __init__(self,
                 d_model: int=256,
                 nhead: int=4, 
                 d_ffn: int=1024,
                 nselflayer: int = 3,
                 ncrosslayer: int = 3,
                 dropout: float=0.1,
                 batch_first: bool=True,
                 activation: str='relu',
                 concat: str='linear',
                 device: str='cuda',
                 skip: bool=True
                 ) -> None:
        super().__init__()
        self.device = device
        
        self.SelfLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            activation=activation,
            batch_first=batch_first,
            device=device
        )        
        self.LeftSelfLayers = nn.ModuleList([deepcopy(self.SelfLayer)
                                             for i in range(nselflayer)])
        self.RightSelfLayers = nn.ModuleList([deepcopy(self.SelfLayer)
                                             for i in range(nselflayer)])
        
        self.CrossLayer = CoTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            device=device,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        self.InputCrossTransformer = nn.ModuleList([deepcopy(self.CrossLayer)
                                               for i in range(ncrosslayer//2)])
        
        self.MidCrossTransformer = deepcopy(self.CrossLayer)
        
        self.OutputCrossTransformer = nn.ModuleList([deepcopy(self.CrossLayer)
                                               for i in range(ncrosslayer//2)])
        self.LeftLinearCatLayers = nn.ModuleList([nn.Linear(2*d_model, d_model)]*(ncrosslayer//2)).to(self.device)
        self.RightLinearCatLayers = nn.ModuleList([nn.Linear(2*d_model, d_model)]*(ncrosslayer//2)).to(self.device)

        self.concat = concat
        self.ncrosslayer = ncrosslayer
        
        if self.concat == 'linear':
            self.OutputLayer = nn.Linear(2*d_model, d_model).to(self.device)
        elif self.concat == 'trunc':
            self.OutputLayer = deepcopy(self.SelfLayer)
        elif self.concat == None:
            self.OutputLayer = None
            
    def forward(self,
                left: Tensor, 
                right:Tensor,
                left_mask:Tensor=None,
                right_mask:Tensor=None,
                left_attn_mask:Tensor=None,
                right_attn_mask:Tensor=None)->Tensor:
        for layer in self.LeftSelfLayers:
            layer:TransformerEncoderLayer
            left = layer.forward(src=left, src_key_padding_mask=left_mask, 
                                 src_mask=left_attn_mask)
        
        for layer in self.RightSelfLayers:
            right = layer.forward(src=right, src_key_padding_mask=right_mask,
                                 src_mask=right_attn_mask)
        
        skip_res = []
        
        for layer in self.InputCrossTransformer:
            layer:CoTransformerLayer
            left, right = layer.forward(
                left = left,
                right = right,
                left_mask=left_mask,
                right_mask=right_mask,
                left_attn_mask=left_attn_mask,
                right_attn_mask=right_attn_mask
            )
            skip_res.append((left, right))

        if self.ncrosslayer > 0:
            left, right = self.MidCrossTransformer.forward(
                left = left, 
                right = right,
                left_mask=left_mask,
                right_mask=right_mask,
                left_attn_mask=left_attn_mask,
                right_attn_mask=right_attn_mask
            )
        
        for i in range(len(self.OutputCrossTransformer)):
            left_skip, right_skip = skip_res.pop()
            left = torch.cat([left, left_skip], dim=-1)
            right = torch.cat([right, right_skip], dim=-1)
            left = self.LeftLinearCatLayers[i].forward(left)
            right = self.RightLinearCatLayers[i].forward(right)
            
            
            left, right = self.OutputCrossTransformer[i].forward(
                left = left,
                right = right,
                left_mask=left_mask,
                right_mask=right_mask,
                left_attn_mask=left_attn_mask,
                right_attn_mask=right_attn_mask
            )    
            

        if self.concat == 'linear':
            output = torch.cat([left, right], dim=-1)
            output = self.OutputLayer(output)
        elif self.concat == 'trunc':
            output = torch.cat([left, right], dim=1)
            output = self.OutputLayer(output)[:,:left.shape[1],:]
        elif self.concat == None:
            output = (left, right)        
        else:
            raise ValueError(f"Invalid concat type: {self.concat}")
        return output 
        
class SkipCoTransformer(nn.Module):
    def __init__(self,
                 d_model: int=256,
                 nhead: int=4, 
                 d_ffn: int=1024,
                 nlayer: int = 3,
                 dropout: float=0.1,
                 batch_first: bool=True,
                 activation: str='relu',
                 device: str='cuda'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_ffn = d_ffn
        self.nlayer = nlayer
        self.dropout = dropout
        self.batch_first = batch_first
        self.activation = activation
        self.device = device
        
        self.co_transformer_layer = CoTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        assert nlayer % 2 == 1
        self.nlayer = (nlayer-1)//2
        
        self.in_blocks = nn.ModuleList([deepcopy(self.co_transformer_layer) for _ in range(self.nlayer)])
        self.mid_blocks = nn.ModuleList([deepcopy(self.co_transformer_layer) for _ in range(1)])
        self.out_blocks = nn.ModuleList([deepcopy(self.co_transformer_layer) for _ in range(self.nlayer)])
        
        self.left_concat_layers = nn.ModuleList([nn.Linear(2*d_model, d_model)]*(self.nlayer)).to(self.device)
        self.right_concat_layers = nn.ModuleList([nn.Linear(2*d_model, d_model)]*(self.nlayer)).to(self.device)
        
    def forward(self,
                left: Tensor,
                right: Tensor,
                left_mask: Tensor=None,
                right_mask: Tensor=None,
                left_attn_mask: Tensor=None,
                right_attn_mask: Tensor=None):
        
        skip_res = []
        
        for layer in self.in_blocks:
            left, right = layer.forward(
                left = left,
                right = right,
                left_mask=left_mask,
                right_mask=right_mask,
                left_attn_mask=left_attn_mask,
                right_attn_mask=right_attn_mask
            )
            skip_res.append((left, right))
        
        left, right = self.mid_blocks[0].forward(
            left = left, 
            right = right,  
            left_mask=left_mask,
            right_mask=right_mask,
            left_attn_mask=left_attn_mask,
            right_attn_mask=right_attn_mask
        )
        
        for i in range(len(self.out_blocks)):
            left_skip, right_skip = skip_res.pop()
            left = torch.cat([left, left_skip], dim=-1)
            right = torch.cat([right, right_skip], dim=-1)
            left = self.left_concat_layers[i].forward(left)
            right = self.right_concat_layers[i].forward(right)
            
            
            left, right = self.out_blocks[i].forward(
                left = left,
                right = right,
                left_mask=left_mask,
                right_mask=right_mask,
                left_attn_mask=left_attn_mask,
                right_attn_mask=right_attn_mask
            )

        return left, right
        
