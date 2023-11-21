import torch
from torch import nn
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
import pickle

__all__ = ['DocParseNet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
from pytesseract import pytesseract
from PIL import Image
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import logging

from multiprocessing import Pool
import re

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class Mish(nn.Module):
    def forward(self, input):
        return input * torch.tanh(F.softplus(input))
    
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = Mish()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=Mish(), drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ImageTextEmbeddings:
    def __init__(self):
        # Initializing the models/tokenizers can be costly. Doing it once in the constructor can be efficient.
        self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # bert-base-uncased, distilbert-base-uncased
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")  # bert-base-uncased
        self.sentence_transformer = SentenceTransformer('bert-base-nli-mean-tokens')
        # self.cache = {}
        self.cache_file = 'embedding00_cache000.pkl'
        if not os.path.exists(self.cache_file):
            self.cache = {}  
        else:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)

    # Convert the image to text using pytesseract
    def image_to_text(self, img_path):
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        return self._clean_text(text)

    # Get sentence embedding using BERT's [CLS] token
    def get_sentence_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding

    # Helper method to clean up the text
    def _clean_text(self, text):
        filtered_text = re.findall(r"\b\w{2,}\b", text)
        filtered_text = ' '.join(filtered_text)
        return filtered_text

    def get_embedding(self, image_id):

        if image_id not in self.cache:
            text_from_image = self.image_to_text(image_id)
            embedding = self.get_sentence_embedding(text_from_image)
            self.cache[image_id] = embedding

            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)

        return self.cache[image_id]

class SpatialAttentionModule(nn.Module):
    def __init__(self, embed_dim, intermediate_dim, spatial_dim):
        super(SpatialAttentionModule, self).__init__()
        
        # Define the attention network
        # This network will transform the text feature to a spatial attention map
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, spatial_dim * spatial_dim),
            nn.Sigmoid()
        )
        
    def forward(self, embed_ids, out):
        # Generate attention map from embed_ids
        attention_map = self.attention_net(embed_ids)  # Shape: [64, 32*32]
        attention_map = attention_map.view(out.size(0), 1, out.size(2), out.size(3))  # Reshape to [64, 1, 32, 32]
        
        # Apply attention to out
        out_with_attention = attention_map * out
        
        return out_with_attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]   #value: torch.Size([64, 256, 768]), key: torch.Size([64, 256, 256]), Query: torch.Size([64, 256, 256])

        # Split the embedding dimension into heads
        query = self.query(query).view(N, query_len, self.num_heads, self.head_dim) #torch.Size([64, 256, 1, 256])
        key = self.key(key).view(N, key_len, self.num_heads, self.head_dim) #torch.Size([64, 256, 1, 256])
        value = self.value(value).view(N, value_len, self.num_heads, self.head_dim) #torch.Size([64, 256, 1, 256])

        # Multi-head attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        attention = torch.nn.functional.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(N, query_len, self.d_model)

        return self.fc_out(out)


class DocParseNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[16, 32, 64, 128, 256, 320],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, embed_dims[0], 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn2 = nn.BatchNorm2d(embed_dims[1])
        self.ebn3 = nn.BatchNorm2d(embed_dims[2])
        
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        self.block3 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        self.dblock0 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        



        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[5])
        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 3, stride=1,padding=1)  
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(embed_dims[2], embed_dims[1], 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(embed_dims[1], embed_dims[0], 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(embed_dims[0], embed_dims[0], 3, stride=1, padding=1)

        self.dbn0 = nn.BatchNorm2d(embed_dims[4])
        self.dbn1 = nn.BatchNorm2d(embed_dims[3])
        self.dbn2 = nn.BatchNorm2d(embed_dims[2])
        self.dbn3 = nn.BatchNorm2d(embed_dims[1])
        self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        
        self.final = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))
            
    def forward(self, x, img_id):        
        B = x.shape[0]

        ### Encoder
        ### Conv Stage

        embed_ids = []        
        processor = ImageTextEmbeddings()
        
        
        for id in img_id:  # "id" size = Batch size
            embedding = processor.get_embedding(id)  
            embed_ids.extend(embedding)
        embed_ids = torch.vstack(embed_ids) 


        ### Stage 1
        out = self.mish(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2)) #shape.x: torch.Size([64, 3, 1024, 1024])
        t1 = out #torch.Size([64, 16, 512, 512])
        
        ### Stage 2
        out = self.mish(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out #torch.Size([64, 32, 256, 256])
        
        ### Stage 3
        out = self.mish(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out #torch.Size([64, 128, 128, 128])

        ### Tokenized MLP Stage
        ### Stage 4
        out,H,W = self.patch_embed3(out) #out:torch.Size([64, 4096, 160]) H: 64   W: 64    

        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out #torch.Size([64, 160, 64, 64])

        ### Stage 5
        out ,H,W= self.patch_embed4(out)  #torch.Size([64, 1024, 256])   H: 32   W: 32
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)   #torch.Size([64, 1024, 256])

        out = self.norm4(out)   #torch.Size([64, 1024, 256])

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  
        t5 = out    #torch.Size([64, 256, 32, 32])

        
        ### Concat part
        B, C, H, W = out.shape #torch.Size([64, 256, 16, 16]) B: 64, H: 32, W: 32, C:256
        out = out.to('cuda:0')
        embed_ids = embed_ids.to('cuda:0') #torch.Size([64, 768])


        ### Expand embed_ids to have spatial dimensions
        embed_expanded = embed_ids.unsqueeze(-1).unsqueeze(-1).expand(B, embed_ids.shape[1], H, W)  # torch.Size([6, 768, 16, 16]) # for 1024/// torch.Size([6, 768, 32, 32])
        embed_expanded = embed_expanded.permute(0, 2, 3, 1).reshape(B, H*W, -1) # Reshape to B x HW x C #format  torch.Size([6, 256, 768]) # for 1024/// torch.Size([6, 1024, 768])

        linear_layer = nn.Linear(768, 256).to('cuda:0')
        embed_expanded = linear_layer(embed_expanded) #torch.Size([6, 256, 256])    #for 1024 torch.Size([6, 1024, 256])

        out = out.flatten(2).transpose(1,2) # Flatten spatial dims and switch to B x HW x C #torch.Size([6, 256, 256])   #for 1024 torch.Size([6, 1024, 256])
        
        ### Multi-head attention
        attention_layer = MultiHeadAttention(d_model=C, num_heads=1).to('cuda:0') # You can adjust the number of heads  #torch.Size([64, 256, 16, 16])     
        out = attention_layer(out, out, embed_expanded) # Using out as Q, K and embed_expanded as V

        ### Reshape to B x C x H x W format
        out = out.transpose(1,2).reshape(B, C, H, W)


        ### Bottleneck
        out, H, W = self.patch_embed5(out) #torch.Size([64, 256, 512])
        for i, blk in enumerate(self.block3): 
            out = blk(out, H, W)  #torch.Size([64, 256, 512])
        out = self.norm5(out)  # torch.Size([64, 256, 512])
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #torch.Size([64, 512, 16, 16])
        
        ### Decoder
        ### Stage 5
        out = self.mish(F.interpolate(self.dbn0(self.decoder0(out)),scale_factor=(2,2),mode ='bilinear'))  #torch.Size([64, 256, 32, 32])
        out = torch.add(out,t5) #torch.Size([64, 256, 32, 32])
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2) #torch.Size([64, 1024, 256])
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)  #torch.Size([64, 1024, 256])
        

        ### Stage 4
        out = self.dnorm2(out) #torch.Size([64, 1024, 256])
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #torch.Size([64, 256, 32, 32])
        out = self.mish(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))  #torch.Size([64, 160, 64, 64])
        out = torch.add(out,t4) #torch.Size([64, 160, 64, 64])
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2) #torch.Size([64, 4096, 160])
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3       
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.mish(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        ### Stage 2
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.mish(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        
        ### Stage 1
        out = self.mish(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = self.mish(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)
        
    