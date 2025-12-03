import torch
import torch.nn as nn
from MobileNetv3.model import mobilenet_v3_large
from my_model.generator import generator_new, generator_woCom
from kappamodules.vit import VitClassTokens
from my_model.my_xlstm.xlstm_lm_model import xLSTMLMModelConfig, xLSTMLMModel
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from RNN.RNN_model import my_rnn
from tsai.models.ConvTranPlus import ConvTranPlus
from ConvTansPlus.convtran_train import my_model as conv_my
from ResNet.resnet_train import TwoChannelInputResNet
from ResNet.model import resnet18, resnet34
from tsai.models.TCN import TCN
from tsai.models.TSSequencerPlus import TSSequencerPlus


class my_model(nn.Module):
    def __init__(self, model, exp_channel):
        super(my_model, self).__init__()

        self.channel_conv = nn.Conv2d(in_channels=2, out_channels=exp_channel, kernel_size=3, padding=1)
        self.model = model

    def forward(self, x):
        # x: [B, 2, H, W]
        x = self.channel_conv(x)  # [B, 3, H, W]
        out = self.model(x)
        return out

class xlstm_cls(nn.Module):
    def __init__(self,model, exp_channel):
        super(xlstm_cls,self).__init__()
        self.channel_conv = nn.Conv2d(in_channels=2, out_channels=exp_channel, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(201, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 4)
        )

        self.cls_token = VitClassTokens(dim=201, num_tokens=1, location="last")
        self.model = model

    def forward(self, x):
        # x: [B, 2, H, W]
        x = self.channel_conv(x)
        x = torch.squeeze(x, 1)

        x = self.cls_token(x)
        out = self.model(x)
        class_token = out[:, -1, :]
        out = self.fc(class_token)
        return out




def load_model(modelname):
    DEVICE = torch.device("cuda:0")
    if modelname == "my_model":
        sample_rate = 8000
        crop_len = 1600
        hop_size = 32
        weight_path = r""
        model = generator_new([4, 4, 8, 2, 2], in_channels=2, in_length=(crop_len // hop_size) + 1, feature_dim=201).to(
            DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))


    elif modelname == "my_model_woCom":
        sample_rate = 8000
        crop_len = 1600
        hop_size = 32
        weight_path = r""
        model = generator_woCom([4, 4, 8, 2, 2], in_channels=2, in_length=(crop_len // hop_size) + 1, feature_dim=201).to(
            DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))



    elif modelname == "MobileNetv3":

        weight_path = ".\MobileNetv3\weight\epoch_60.pth"
        ori_model = mobilenet_v3_large(num_classes=4)
        model = my_model(ori_model, 3).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))



    elif modelname == "xLSTM":
        weight_path = r".\xlstm\weight\epoch_60.pth"
        time_xlstm_cfg_init = f"""
        context_length: 52      
        num_blocks: 12
        embedding_dim: 201
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
        cfg = OmegaConf.create(time_xlstm_cfg_init)
        cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))

        ori_model = xLSTMLMModel(cfg)
        model = xlstm_cls(ori_model,1).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "LSTM":
        crop_len = 1600
        hop_size = 32
        weight_path = r".\RNN\LSTM_weight\epoch_60.pth"
        model = my_rnn([4, 4, 8, 2, 2],
                            in_channels=2,
                            in_length=(crop_len // hop_size) + 1,
                            feature_dim=201,
                            baseline= modelname).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "RNN":
        crop_len = 1600
        hop_size = 32
        weight_path = r".\RNN\RNN_weight\epoch_60.pth"
        model = my_rnn([4, 4, 8, 2, 2],
                            in_channels=2,
                            in_length=(crop_len // hop_size) + 1,
                            feature_dim=201,
                            baseline= modelname).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "GRU":
        crop_len = 1600
        hop_size = 32
        weight_path = r".\RNN\GRU_weight\epoch_60.pth"
        model = my_rnn([4, 4, 8, 2, 2],
                            in_channels=2,
                            in_length=(crop_len // hop_size) + 1,
                            feature_dim=201,
                            baseline= modelname).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "ConvTransPlus":
        weight_path = r".\ConvTansPlus\weights\epoch_60.pth"
        ori_model = ConvTranPlus(201,201,51,d=None,d_model=64,dim_ff=512)
        model = conv_my(ori_model,1).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "Resnet18":
        weight_path = r".\ResNet\weight_18\epoch_60.pth"
        ori_model = resnet18(4)
        model = TwoChannelInputResNet(ori_model).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "Resnet34":
        weight_path = r".\ResNet\weight_34\epoch_60.pth"
        ori_model = resnet34(4)
        model = TwoChannelInputResNet(ori_model).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "TCN":
        weight_path = r".\TCN\weights\epoch_60.pth"
        ori_model = TCN(201,201,layers=[256,256,256,128,128,128,64,64,64],ks=3,conv_dropout=0.2,fc_dropout=0.2).to(device=DEVICE)
        model = conv_my(ori_model,1).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    elif modelname == "TSSequencer":
        weight_path = r".\TSSequencer\weights\epoch_60.pth"
        ori_model = TSSequencerPlus(201,201,51,d_model=512,depth=6,lstm_dropout=0.1,dropout=0.1,mlp_ratio=2,
                        pre_norm=True,use_token=True,fc_dropout=0.1).to(device=DEVICE)
        model = conv_my(ori_model,1).to(device=DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))



    else:
        raise

    return model
