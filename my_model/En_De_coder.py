import torch
import torch.nn as nn
from my_model.my_xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from my_model.utils import LearnableSigmoid2d
from my_model.dynamic_tanh import DynamicTanh
from kappamodules.vit import VitClassTokens
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Bi_mlstm(nn.Module):
    # 已自带layernormal
    def __init__(self, mlstm_cfg, hidden_size):
        super(Bi_mlstm, self).__init__()
        self.FRNN = xLSTMLMModel(mlstm_cfg)
        self.BRNN = xLSTMLMModel(mlstm_cfg)
        self.down_conv = nn.Conv1d(2 * hidden_size, hidden_size, kernel_size=1)

    def forward(self, x, enc_output=None):
        y = x
        if enc_output is not None:
            x = self.FRNN(x, enc_output)
            y = self.BRNN(y.flip(dims=[1]), enc_output.flip(dims=[1]))
        else:
            x = self.FRNN(x)
            y = self.BRNN(y.flip(dims=[1]))

        x = torch.cat([x, y], -1)
        x = self.down_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class mlstm(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size):
        super(mlstm, self).__init__()
        self.mlstm_blocks = xLSTMLMModel(mlstm_cfg)

    def forward(self,x):
        x = self.mlstm_blocks(x)
        return x

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mlstm = mlstm(mlstm_cfg, hidden_size)

        self.Norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.DyT = DynamicTanh(hidden_size,channels_last=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)

    def forward(self, x):

        y = self.mlstm(x)
        y = self.dropout(y)
        x = x + y

        y = self.DyT(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.mlstm = mlstm(mlstm_cfg, hidden_size)
        self.enc_dec_mlstm = Bi_mlstm(mlstm_cfg, hidden_size)
        self.Norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.DyT = DynamicTanh(hidden_size,channels_last=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)

    def forward(self, x, enc_output=None):

        y = self.mlstm(x)
        y = self.dropout(y)
        x = x + y

        if enc_output is not None:
            y = self.Norm(x)
            y = self.enc_dec_mlstm(y, enc_output)
            y = self.dropout(y)
            x = x + y

        # y = self.Norm(x)
        y = self.DyT(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate, n_layers):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(mlstm_cfg, hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs):

        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output)
        return self.last_norm(encoder_output)


class Decoder(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(mlstm_cfg, hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output=None):
        decoder_output = targets
        for dec_layer in self.layers:
            decoder_output = dec_layer(decoder_output, enc_output)
        return self.last_norm(decoder_output)


class MaskDecoder(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate, n_layers):
        super(MaskDecoder, self).__init__()
        decoders = [DecoderLayer(mlstm_cfg, hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.transformer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList(decoders)
        self.sub_pixel = SPConvTranspose2d(1, 1, (1, 3), 4)
        self.conv_1 = nn.Conv2d(1, 1, (1, 2), (1, 1), (0, 1))
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.prelu = nn.PReLU(1)
        self.final_conv = nn.Conv2d(1, 1, (1, 1))
        # self.prelu_out = nn.PReLU((4*hidden_size) + 1, init=-0.25)
        self.lsigmoid = LearnableSigmoid2d((4*hidden_size) + 1, beta=2)

    def forward(self, x):
        for dec_layer in self.layers:
            x = dec_layer(x)    #(b,t,f)
        x = self.transformer_norm(x).unsqueeze(1)    # (b,c,t,f)
        x = self.sub_pixel(x)   # (b,c,t,r*f')
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)  # (b,f,t)
        # return self.prelu_out(x)
        return self.lsigmoid(x)

class PhaseDecoder(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate, n_layers):
        super(PhaseDecoder, self).__init__()
        decoders = [DecoderLayer(mlstm_cfg, hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.transformer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList(decoders)
        self.sub_pixel = SPConvTranspose2d(1, 1, (1, 3), 4)
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.prelu = nn.PReLU(1)
        self.phase_conv_r = nn.Conv2d(1, 1, (1, 2), (1, 1), (0, 1))
        self.phase_conv_i = nn.Conv2d(1, 1, (1, 2), (1, 1), (0, 1))

    def forward(self, x):
        for dec_layer in self.layers:
            x = dec_layer(x)
        x = self.transformer_norm(x).unsqueeze(1)     # (b,c,t,f)
        x = self.sub_pixel(x)   # (b,c,t,r*f')
        x = self.prelu(self.norm(x))
        x_r = self.phase_conv_r(x).permute(0, 3, 2, 1).squeeze(-1)
        x_i = self.phase_conv_i(x).permute(0, 3, 2, 1).squeeze(-1)
        x = torch.atan2(x_i, x_r)
        return x


class ComplexDecoder(nn.Module):
    def __init__(self, mlstm_cfg, hidden_size, filter_size, dropout_rate, n_layers):
        super(ComplexDecoder, self).__init__()
        decoders = [DecoderLayer(mlstm_cfg, hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.transformer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList(decoders)
        self.sub_pixel = SPConvTranspose2d(1, 1, (1, 3), 4)
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.prelu = nn.PReLU(1)
        self.complex_conv = nn.Conv2d(1, 2, (1, 2),(1, 1), (0, 1))

    def forward(self, x):
        for dec_layer in self.layers:
            x = dec_layer(x)
        x = self.transformer_norm(x).unsqueeze(1)     # (b,c,t,f)
        x = self.sub_pixel(x)   # (b,c,t,r*f')
        x = self.prelu(self.norm(x))
        x = self.complex_conv(x)
        return x.permute(0, 1, 3, 2)


class Classifier(nn.Module):
    def __init__(self,mlstm_cfg,hidden_size,num_classes):
        super(Classifier, self).__init__()
        self.mlstm = mlstm(mlstm_cfg, hidden_size)
        self.cls_token = VitClassTokens(dim=hidden_size, num_tokens=1,location="last")
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_classes)
        )

    def forward(self,x):
        x = self.cls_token(x)
        x = self.mlstm(x)
        class_token = x[:,-1,:]
        y = self.fc(class_token)
        return  y

class Classifier_new(nn.Module):
    def __init__(self,in_channels,num_blocks, in_length, in_feature,num_classes):
        super(Classifier_new, self).__init__()

        classifier_xlstm_cfg_init = f"""
        context_length: {in_length +1}      
        num_blocks: {num_blocks}
        embedding_dim: {in_feature} 
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
        mlstm_cfg = OmegaConf.create(classifier_xlstm_cfg_init)
        mlstm_cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(mlstm_cfg),
                        config=DaciteConfig(strict=True))

        self.conv = nn.Sequential(
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
            nn.Conv2d(32, 1, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1),
        )
        self.mlstm = mlstm(mlstm_cfg, in_feature)
        self.cls_token = VitClassTokens(dim=in_feature, num_tokens=1,location="last")
        self.fc = nn.Sequential(
            nn.LayerNorm(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature,num_classes)
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze(1)
        x = self.cls_token(x)
        x = self.mlstm(x)
        class_token = x[:,-1,:]
        y = self.fc(class_token)
        return y


def main():
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.rand(4, 2, 51, 201).to(device=DEVICE)
    # model = generator(device=DEVICE,in_channels=2).to(device=DEVICE)
    model = Classifier_new(2,4,51,50,4).to(device=DEVICE)
    est = model(x)
    print(est.shape)
    # print(model)
    # print(f"Model Size: {get_model_size(model):.2f} MB")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


if __name__ == '__main__':
    main()
