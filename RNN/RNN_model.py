import torch
import torch.nn as nn
from RNN.en_decoder import Encoder,MaskDecoder,ComplexDecoder,Classifier
from RNN.conformer import TSCB


class my_rnn(nn.Module):
    def __init__(self, layer=[2, 4, 3, 3, 3], in_channels=2, in_length=51, feature_dim=201, baseline='RNN'):
        super(my_rnn, self).__init__()
        
        self.xlstm_blocknum = layer[0]
        self.in_length = in_length
        self.feature_dim = feature_dim // 4
        self.TSCB_layer_num = layer[1]

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
        self.Encoder = Encoder(self.feature_dim,
                             4 * self.feature_dim,
                             dropout_rate=0.1,
                             n_layers=layer[2],baseline=baseline)
        TSCBs = [TSCB(self.in_length, self.feature_dim,baseline=baseline)
                 for _ in range(self.TSCB_layer_num)]
        self.TSCB = nn.ModuleList(TSCBs)


        self.Mask_Decoder = MaskDecoder(self.feature_dim,
                                         4 * self.feature_dim,
                                         dropout_rate=0.05,
                                         n_layers=layer[3],baseline=baseline)
        self.Com_Decoder = ComplexDecoder(self.feature_dim,
                                           4 * self.feature_dim,
                                           dropout_rate=0.05,
                                           n_layers=layer[4],baseline=baseline)

        self.Classifier = Classifier(self.feature_dim,
                                        num_classes=4,baseline=baseline)



    def forward(self, x):  # input:(b,c,t,f) c1:mag c2:pha
        noisy_mag, noisy_pha = x[:, 0, :, :].permute(0,2,1), x[:, 1, :, :].permute(0,2,1)   #(b,f,t),此处是为了和mask等形状匹配
        x = self.cov_head(x)
        x = torch.squeeze(x, 1) #(b,t,f')
        x = self.Encoder(x)
        for tscb_layer in self.TSCB: # (batch,time,freq)
            x = tscb_layer(x)   # (batch,time,freq)
        class_outputs = self.Classifier(x)
        decoder_x = x


        mask = self.Mask_Decoder(decoder_x) # (batch,f,t)
        out_mag = mask * noisy_mag

        out_com = self.Com_Decoder(decoder_x)
        mag_real = out_mag * torch.cos(noisy_pha)
        mag_imag = out_mag * torch.sin(noisy_pha)
        final_real = mag_real + out_com[:, 0, :, :]
        final_imag = mag_imag + out_com[:, 1, :, :]

        return final_real, final_imag, class_outputs

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.rand(2,2,51,201).to(DEVICE)
    model = my_rnn(layer=[0, 4, 8, 2, 2], in_channels=2, in_length=51, feature_dim=201).to(DEVICE)
    y1,_,cls = model(x)
    print(cls.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")