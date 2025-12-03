import my_model.En_De_coder
import my_model.conformer
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from my_model.my_xlstm.xlstm_lm_model import xLSTMLMModelConfig


class generator_new(nn.Module):
    def __init__(self, layer=[2, 4, 3, 3, 3], in_channels=2, in_length=101, feature_dim=201):
        super(generator_new, self).__init__()

        self.xlstm_blocknum = layer[0]
        self.in_length = in_length
        self.feature_dim = feature_dim // 4
        self.TSCB_layer_num = layer[1]
        time_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        freq_xlstm_cfg_init = f"""
        context_length: {self.feature_dim}    
        num_blocks: {self.xlstm_blocknum}  
        embedding_dim: {self.in_length}   
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        decoder_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: true
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        classifier_xlstm_cfg_init = f"""
        context_length: {self.in_length +1}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: true
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        time_cfg = self.cfg_set(time_xlstm_cfg_init)
        freq_cfg = self.cfg_set(freq_xlstm_cfg_init)
        dec_cfg = self.cfg_set(decoder_xlstm_cfg_init)
        cls_cfg = self.cfg_set(classifier_xlstm_cfg_init)
        self.cov_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128),
            nn.Conv2d(128, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1)
        )

        self.Encoder = my_model.En_De_coder.Encoder(time_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[2])
        TSCBs = [my_model.conformer.TSCB(self.in_length, self.feature_dim, time_cfg, freq_cfg)
                 for _ in range(self.TSCB_layer_num)]
        self.TSCB = nn.ModuleList(TSCBs)
        self.Mask_Decoder = my_model.En_De_coder.MaskDecoder(dec_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[3])
        self.Com_Decoder = my_model.En_De_coder.ComplexDecoder(dec_cfg,
                                                               self.feature_dim,
                                                               4 * self.feature_dim,
                                                               dropout_rate=0.1,
                                                               n_layers=layer[4])
        self.Classifier = my_model.En_De_coder.Classifier(cls_cfg,self.feature_dim,4)

    def cfg_set(self, xlstm_cfg):
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))
        return cfg

    def forward(self, x, detach_input=False):  # input:(b,c,t,f) c1:mag c2:pha
        noisy_mag, noisy_pha = x[:, 0, :, :].permute(0,2,1), x[:, 1, :, :].permute(0,2,1)   #(b,f,t),此处是为了和mask等形状匹配
        x = self.cov_head(x)
        x = torch.squeeze(x, 1) #(b,t,f')
        x = self.Encoder(x)
        for tscb_layer in self.TSCB: # (batch,time,freq)
            x = tscb_layer(x)   # (batch,time,freq)
        class_outputs = self.Classifier(x)

        # if detach_input:
        #     decoder_x = x.detach()
        # else:
        decoder_x = x

        mask = self.Mask_Decoder(decoder_x) # (batch,f,t)
        out_mag = mask * noisy_mag

        out_com = self.Com_Decoder(decoder_x)
        mag_real = out_mag * torch.cos(noisy_pha)
        mag_imag = out_mag * torch.sin(noisy_pha)
        final_real = mag_real + out_com[:, 0, :, :]
        final_imag = mag_imag + out_com[:, 1, :, :]

        return final_real, final_imag, class_outputs

class generator_withoutcls(nn.Module):
    def __init__(self, layer=[2, 4, 3, 3, 3], in_channels=2, in_length=101, feature_dim=201):
        super(generator_withoutcls, self).__init__()

        self.xlstm_blocknum = layer[0]
        self.in_length = in_length
        self.feature_dim = feature_dim // 4
        self.TSCB_layer_num = layer[1]
        time_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        freq_xlstm_cfg_init = f"""
        context_length: {self.feature_dim}    
        num_blocks: {self.xlstm_blocknum}  
        embedding_dim: {self.in_length}   
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        decoder_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: true
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """

        time_cfg = self.cfg_set(time_xlstm_cfg_init)
        freq_cfg = self.cfg_set(freq_xlstm_cfg_init)
        dec_cfg = self.cfg_set(decoder_xlstm_cfg_init)
        self.cov_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128),
            nn.Conv2d(128, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1)
        )
        self.Encoder = my_model.En_De_coder.Encoder(time_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[2])
        TSCBs = [my_model.conformer.TSCB(self.in_length, self.feature_dim, time_cfg, freq_cfg)
                 for _ in range(self.TSCB_layer_num)]
        self.TSCB = nn.ModuleList(TSCBs)
        self.Mask_Decoder = my_model.En_De_coder.MaskDecoder(dec_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[3])
        self.Com_Decoder = my_model.En_De_coder.ComplexDecoder(dec_cfg,
                                                               self.feature_dim,
                                                               4 * self.feature_dim,
                                                               dropout_rate=0.1,
                                                               n_layers=layer[4])

    def cfg_set(self, xlstm_cfg):
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))
        return cfg

    def forward(self, x):  # input:(b,c,t,f) c1:mag c2:pha
        noisy_mag, noisy_pha = x[:, 0, :, :].permute(0,2,1), x[:, 1, :, :].permute(0,2,1)   #(b,f,t),此处是为了和mask等形状匹配
        x = self.cov_head(x)
        x = torch.squeeze(x, 1) #(b,t,f')
        x = self.Encoder(x)
        for tscb_layer in self.TSCB: # (batch,time,freq)
            x = tscb_layer(x)   # (batch,time,freq)
        mask = self.Mask_Decoder(x) # (batch,f,t)
        out_mag = mask * noisy_mag

        out_com = self.Com_Decoder(x)
        mag_real = out_mag * torch.cos(noisy_pha)
        mag_imag = out_mag * torch.sin(noisy_pha)
        final_real = mag_real + out_com[:, 0, :, :]
        final_imag = mag_imag + out_com[:, 1, :, :]

        return final_real, final_imag

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # 计算参数占用的字节数
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # 计算缓冲区占用的字节数
    total_size = (param_size + buffer_size) / (1024 ** 2)  # 转换为 MB
    return total_size


class Classifier_withtscb(nn.Module):
    def __init__(self, layer=[2, 4, 3, 3, 3], in_channels=2, in_length=51, feature_dim=201):
        super(Classifier_withtscb, self).__init__()

        self.xlstm_blocknum = layer[0]
        self.in_length = in_length
        self.feature_dim = feature_dim // 4
        self.TSCB_layer_num = layer[1]
        time_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        freq_xlstm_cfg_init = f"""
        context_length: {self.feature_dim}    
        num_blocks: {self.xlstm_blocknum}  
        embedding_dim: {self.in_length}   
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        cls_xlstm_cfg_init = f"""
        context_length: {self.in_length+1}      
        num_blocks: {self.xlstm_blocknum*3} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: true
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
                """
        time_cfg = self.cfg_set(time_xlstm_cfg_init)
        freq_cfg = self.cfg_set(freq_xlstm_cfg_init)
        cls_cfg = self.cfg_set(cls_xlstm_cfg_init)
        self.cov_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128),
            nn.Conv2d(128, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1)
        )
        self.Encoder = my_model.En_De_coder.Encoder(time_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[2])
        TSCBs = [my_model.conformer.TSCB(self.in_length, self.feature_dim, time_cfg, freq_cfg)
                 for _ in range(self.TSCB_layer_num)]
        self.TSCB = nn.ModuleList(TSCBs)
        self.cls_head = my_model.En_De_coder.Classifier(cls_cfg,self.feature_dim,4)


    def cfg_set(self, xlstm_cfg):
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))
        return cfg

    def forward(self, x):  # input:(b,c,t,f) c1:mag c2:pha

        x = self.cov_head(x)
        x = x.squeeze(1)
        x = self.Encoder(x)
        for tscb_layer in self.TSCB: # (batch,time,freq)
            x = tscb_layer(x)   # (batch,time,freq)
        x = self.cls_head(x)
        return x


class generator_woCom(nn.Module):
    def __init__(self, layer=[2, 4, 3, 3, 3], in_channels=2, in_length=101, feature_dim=201):
        super(generator_woCom, self).__init__()

        self.xlstm_blocknum = layer[0]
        self.in_length = in_length
        self.feature_dim = feature_dim // 4
        self.TSCB_layer_num = layer[1]
        time_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        freq_xlstm_cfg_init = f"""
        context_length: {self.feature_dim}    
        num_blocks: {self.xlstm_blocknum}  
        embedding_dim: {self.in_length}   
        tie_weights: false
        weight_decay_on_embedding: true
        mask: false
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        decoder_xlstm_cfg_init = f"""
        context_length: {self.in_length}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: true
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        classifier_xlstm_cfg_init = f"""
        context_length: {self.in_length +1}      
        num_blocks: {self.xlstm_blocknum} 
        embedding_dim: {self.feature_dim} 
        tie_weights: false
        weight_decay_on_embedding: true
        mask: true
        dropout: 0.1
        mlstm_block:
          mlstm: 
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
        """
        time_cfg = self.cfg_set(time_xlstm_cfg_init)
        freq_cfg = self.cfg_set(freq_xlstm_cfg_init)
        dec_cfg = self.cfg_set(decoder_xlstm_cfg_init)
        cls_cfg = self.cfg_set(classifier_xlstm_cfg_init)
        self.cov_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.PReLU(128),
            nn.Conv2d(128, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(32, affine=True),
            nn.PReLU(32),
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1)
        )
        self.Encoder = my_model.En_De_coder.Encoder(time_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[2])
        TSCBs = [my_model.conformer.TSCB(self.in_length, self.feature_dim, time_cfg, freq_cfg)
                 for _ in range(self.TSCB_layer_num)]
        self.TSCB = nn.ModuleList(TSCBs)
        self.Mask_Decoder = my_model.En_De_coder.MaskDecoder(dec_cfg,
                                                             self.feature_dim,
                                                             4 * self.feature_dim,
                                                             dropout_rate=0.1,
                                                             n_layers=layer[3])

        self.Classifier = my_model.En_De_coder.Classifier(cls_cfg,self.feature_dim,4)

    def cfg_set(self, xlstm_cfg):
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))
        return cfg

    def forward(self, x, detach_input=False):  # input:(b,c,t,f) c1:mag c2:pha
        noisy_mag, noisy_pha = x[:, 0, :, :].permute(0,2,1), x[:, 1, :, :].permute(0,2,1)   #(b,f,t),此处是为了和mask等形状匹配
        x = self.cov_head(x)
        x = torch.squeeze(x, 1) #(b,t,f')
        x = self.Encoder(x)
        for tscb_layer in self.TSCB: # (batch,time,freq)
            x = tscb_layer(x)   # (batch,time,freq)
        class_outputs = self.Classifier(x)

        if detach_input:
            decoder_x = x.detach()
        else:
            decoder_x = x

        mask = self.Mask_Decoder(decoder_x) # (batch,f,t)
        out_mag = mask * noisy_mag

        mag_real = out_mag * torch.cos(noisy_pha)
        mag_imag = out_mag * torch.sin(noisy_pha)


        return mag_real, mag_imag, class_outputs




def main():
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.rand(1, 2, 51, 201).to(device=DEVICE)
    # model = generator(device=DEVICE,in_channels=2).to(device=DEVICE)
    model = generator_new(layer=[2, 4, 8, 2, 2],in_channels=2,in_length=51,feature_dim=201).to(device=DEVICE)
    _,_,y1 = model(x)
    print(y1.shape)
    # print(model)
    # print(f"Model Size: {get_model_size(model):.2f} MB")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


if __name__ == '__main__':
    main()
